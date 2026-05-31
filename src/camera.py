from __future__ import annotations

import cv2
import numpy as np


class CameraStream:
    # Custom wrapper around OpenCV capture so the input source can later change
    # from the laptop webcam to the drone stream without rewriting pipeline code.
    """Thin webcam wrapper that keeps the local input source replaceable."""

    def __init__(self, camera_index: int, frame_width: int, frame_height: int) -> None:
        self._camera_index = camera_index
        self._frame_width = frame_width
        self._frame_height = frame_height
        self._capture: cv2.VideoCapture | None = None

    def open(self) -> None:
        capture = cv2.VideoCapture(self._camera_index)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self._frame_width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self._frame_height)

        if not capture.isOpened():
            capture.release()
            raise RuntimeError(
                f"Unable to open camera index {self._camera_index}."
            )

        self._capture = capture

    def read(self) -> np.ndarray:
        if self._capture is None:
            raise RuntimeError("Camera stream has not been opened.")

        success, frame = self._capture.read()
        if not success or frame is None:
            raise RuntimeError("Failed to read a frame from the camera stream.")

        return frame

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None
