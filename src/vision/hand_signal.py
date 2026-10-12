from __future__ import annotations

import math
import time
from pathlib import Path

import cv2
import numpy as np

try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
except ImportError:  # pragma: no cover - depends on local environment
    mp = None
    mp_python = None
    mp_vision = None


Point = tuple[float, float]


class OpenPalmDetector:
    """Find open palms held upright toward the camera."""

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

    def __init__(self, model_path: str) -> None:
        if mp is None or mp_python is None or mp_vision is None:
            raise RuntimeError(
                "mediapipe is not installed in the active environment. "
                "Install the project requirements to use hand-signal lock-on."
            )

        resolved_model_path = Path(model_path)
        if not resolved_model_path.is_file():
            raise FileNotFoundError(
                f"Hand landmark model not found: {resolved_model_path}. "
                f"Download it with: curl -L {self.MODEL_URL} "
                "-o hand_landmarker.task"
            )

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(resolved_model_path)
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.6,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def detect_open_palms(self, frame_bgr: np.ndarray) -> list[Point]:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        timestamp_ms = max(time.monotonic_ns() // 1_000_000, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._landmarker.detect_for_video(image, timestamp_ms)
        points: list[Point] = []

        if not results.hand_landmarks:
            return points

        frame_height, frame_width = frame_bgr.shape[:2]
        for landmarks in results.hand_landmarks:
            fingers_extended = all(
                landmarks[tip].y < landmarks[pip].y
                for tip, pip in ((8, 6), (12, 10), (16, 14), (20, 18))
            )
            palm_width = math.dist(
                (landmarks[5].x, landmarks[5].y),
                (landmarks[17].x, landmarks[17].y),
            )
            thumb_open = math.dist(
                (landmarks[4].x, landmarks[4].y),
                (landmarks[5].x, landmarks[5].y),
            ) > palm_width * 0.5

            if fingers_extended and thumb_open:
                palm_center_x = (landmarks[5].x + landmarks[17].x) / 2.0
                palm_center_y = (landmarks[5].y + landmarks[17].y) / 2.0
                points.append(
                    (palm_center_x * frame_width, palm_center_y * frame_height)
                )

        return points

    def close(self) -> None:
        self._landmarker.close()


class OpenPalmToggle:
    """Emit one toggle event after an open palm is held for one second."""

    HOLD_SECONDS = 1.0
    RELEASE_SECONDS = 0.25

    def __init__(self) -> None:
        self._hold_started_at: float | None = None
        self._last_seen_at: float | None = None
        self._fired = False

    def update(self, palm_points: list[Point], now: float | None = None) -> Point | None:
        current_time = time.monotonic() if now is None else now
        if palm_points:
            self._last_seen_at = current_time
            if self._hold_started_at is None:
                self._hold_started_at = current_time
            if not self._fired and current_time - self._hold_started_at >= self.HOLD_SECONDS:
                self._fired = True
                return palm_points[0]
            return None

        if (
            self._last_seen_at is not None
            and current_time - self._last_seen_at < self.RELEASE_SECONDS
        ):
            return None

        self._hold_started_at = None
        self._last_seen_at = None
        self._fired = False
        return None
