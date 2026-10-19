from __future__ import annotations

import math
import shutil
import tempfile
import time
from pathlib import Path
from typing import Protocol
from urllib.error import URLError
from urllib.request import urlopen

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
Point3D = tuple[float, float, float]


class Landmark3D(Protocol):
    x: float
    y: float
    z: float


def _distance_3d(first: Point3D, second: Point3D) -> float:
    return math.dist(first, second)


def _joint_angle_degrees(first: Point3D, joint: Point3D, last: Point3D) -> float:
    first_vector = tuple(first[index] - joint[index] for index in range(3))
    last_vector = tuple(last[index] - joint[index] for index in range(3))
    first_length = math.sqrt(sum(value * value for value in first_vector))
    last_length = math.sqrt(sum(value * value for value in last_vector))
    if first_length == 0.0 or last_length == 0.0:
        return 0.0

    cosine = sum(
        first_vector[index] * last_vector[index] for index in range(3)
    ) / (first_length * last_length)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _is_open_palm(world_landmarks: list[Landmark3D]) -> bool:
    points: list[Point3D] = [
        (landmark.x, landmark.y, landmark.z) for landmark in world_landmarks
    ]
    extended_fingers = all(
        _joint_angle_degrees(points[mcp], points[pip], points[dip]) >= 145.0
        and _joint_angle_degrees(points[pip], points[dip], points[tip]) >= 140.0
        and _distance_3d(points[mcp], points[tip])
        > _distance_3d(points[mcp], points[pip]) * 1.1
        for mcp, pip, dip, tip in (
            (5, 6, 7, 8),
            (9, 10, 11, 12),
            (13, 14, 15, 16),
            (17, 18, 19, 20),
        )
    )
    palm_width = _distance_3d(points[5], points[17])
    thumb_open = _distance_3d(points[4], points[5]) > palm_width * 0.5
    return extended_fingers and thumb_open


class OpenPalmDetector:
    """Find open palms held upright toward the camera."""

    MODEL_URL = (
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
        "hand_landmarker/float16/1/hand_landmarker.task"
    )

    @classmethod
    def _ensure_model_file(cls, model_path: Path) -> None:
        if model_path.is_file():
            return

        temporary_path: Path | None = None
        try:
            model_path.parent.mkdir(parents=True, exist_ok=True)
            with urlopen(cls.MODEL_URL, timeout=20) as response:
                with tempfile.NamedTemporaryFile(
                    dir=model_path.parent,
                    prefix=f".{model_path.name}.",
                    suffix=".download",
                    delete=False,
                ) as temporary_file:
                    temporary_path = Path(temporary_file.name)
                    shutil.copyfileobj(response, temporary_file)

            if temporary_path.stat().st_size < 1_000_000:
                raise RuntimeError(
                    f"Downloaded hand landmark model is unexpectedly small: "
                    f"{temporary_path.stat().st_size} bytes."
                )
            temporary_path.replace(model_path)
        except (OSError, URLError) as error:
            raise RuntimeError(
                f"Could not download the hand landmark model to {model_path}: "
                f"{error}. Connect to the internet and rerun, or download it "
                f"before connecting to the Tello Wi-Fi with: curl -fL "
                f"{cls.MODEL_URL} -o {model_path}"
            ) from error
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass

    def __init__(self, model_path: str) -> None:
        if mp is None or mp_python is None or mp_vision is None:
            raise RuntimeError(
                "mediapipe is not installed in the active environment. "
                "Install the project requirements to use hand-signal lock-on."
            )

        resolved_model_path = Path(model_path)
        self._ensure_model_file(resolved_model_path)

        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(resolved_model_path)
            ),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=0.45,
            min_hand_presence_confidence=0.45,
            min_tracking_confidence=0.45,
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
        for index, landmarks in enumerate(results.hand_landmarks):
            if _is_open_palm(results.hand_world_landmarks[index]):
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
    RELEASE_SECONDS = 0.45

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
