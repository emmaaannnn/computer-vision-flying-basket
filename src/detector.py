from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import VisionConfig

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - depends on local environment
    YOLO = None


@dataclass(slots=True)
class PersonDetection:
    # Custom reduced state that decouples the rest of the code from the raw
    # Ultralytics result objects.
    """Reduced person-target state extracted from the detector output."""

    x_center_px: float
    y_center_px: float
    width_px: float
    height_px: float
    confidence: float
    box_xyxy: tuple[int, int, int, int]


class PersonDetector:
    # Custom adapter around the external YOLO model. The model does detection;
    # this repo decides how detections are filtered and which person to track.
    """YOLO wrapper restricted to the person class for local validation."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config

        if not config.enable_inference:
            self._model = None
            return

        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed in the active environment."
            )

        self._model = YOLO(config.model_name)

    def detect_primary_person(self, frame_bgr: np.ndarray) -> PersonDetection | None:
        if self._model is None:
            return None

        # Restrict inference to the COCO "person" class so the downstream
        # pipeline only sees the target category it is built around.
        results = self._model.predict(
            source=frame_bgr,
            classes=[0],
            conf=self._config.confidence_threshold,
            verbose=False,
        )

        if not results:
            return None

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return None

        xyxy_list = boxes.xyxy.cpu().tolist()
        confidence_list = boxes.conf.cpu().tolist()
        # Custom target-selection rule: follow the largest visible person box,
        # which is a simple proxy for the main subject in front of the camera.
        best_index = max(
            range(len(xyxy_list)),
            key=lambda index: self._box_area(xyxy_list[index]),
        )

        x1, y1, x2, y2 = xyxy_list[best_index]
        width_px = max(0.0, x2 - x1)
        height_px = max(0.0, y2 - y1)

        return PersonDetection(
            x_center_px=x1 + (width_px / 2.0),
            y_center_px=y1 + (height_px / 2.0),
            width_px=width_px,
            height_px=height_px,
            confidence=float(confidence_list[best_index]),
            box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
        )

    @staticmethod
    def _box_area(box_xyxy: list[float]) -> float:
        x1, y1, x2, y2 = box_xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)
