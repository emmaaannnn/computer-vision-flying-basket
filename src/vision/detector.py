from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - depends on local environment
    YOLO = None


@dataclass(slots=True)
class PersonDetection:
    """Reduced person-target state extracted from the detector output."""

    x_center_px: float
    y_center_px: float
    width_px: float
    height_px: float
    confidence: float
    box_xyxy: tuple[int, int, int, int]


class PersonDetector:
    """YOLO wrapper restricted to the person class."""

    def __init__(self, model_name: str, confidence_threshold: float = 0.35) -> None:
        if YOLO is None:
            raise RuntimeError(
                "ultralytics is not installed in the active environment."
            )

        self._model = YOLO(model_name)
        self._confidence_threshold = confidence_threshold

    def detect_primary_person(self, frame_bgr: np.ndarray) -> PersonDetection | None:
        detections = self.detect_persons(frame_bgr)
        if not detections:
            return None
        return max(detections, key=lambda detection: detection.width_px * detection.height_px)

    def detect_persons(self, frame_bgr: np.ndarray) -> list[PersonDetection]:
        if self._model is None:
            return []

        results = self._model.predict(
            source=frame_bgr,
            classes=[0],
            conf=self._confidence_threshold,
            verbose=False,
        )

        if not results:
            return []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return []

        xyxy_list = boxes.xyxy.cpu().tolist()
        confidence_list = boxes.conf.cpu().tolist()
        detections: list[PersonDetection] = []
        for box, confidence in zip(xyxy_list, confidence_list):
            x1, y1, x2, y2 = box
            width_px = max(0.0, x2 - x1)
            height_px = max(0.0, y2 - y1)
            detections.append(
                PersonDetection(
                    x_center_px=x1 + (width_px / 2.0),
                    y_center_px=y1 + (height_px / 2.0),
                    width_px=width_px,
                    height_px=height_px,
                    confidence=float(confidence),
                    box_xyxy=(int(x1), int(y1), int(x2), int(y2)),
                )
            )

        return detections
