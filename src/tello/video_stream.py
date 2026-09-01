from __future__ import annotations

import time

import cv2
import numpy as np

from src.tello.controller import TelloController


class TelloVideoStream:
    """Open the Tello camera stream and provide frames to the laptop."""

    def __init__(self) -> None:
        self._tello = TelloController()
        self._frame_read = None

    def open(self) -> None:
        self._tello.connect()
        self._tello.stream_on()
        self._frame_read = self._tello.get_frame_read()
        time.sleep(1.5)

    def read(self) -> np.ndarray:
        if self._frame_read is None:
            raise RuntimeError("Tello video stream has not been opened.")

        frame = self._frame_read.frame
        if frame is None:
            raise RuntimeError("Failed to read a frame from the Tello stream.")

        return frame

    def release(self) -> None:
        self._frame_read = None
        self._tello.end()
