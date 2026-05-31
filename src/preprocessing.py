from __future__ import annotations

import cv2
import numpy as np


def apply_luminance_clahe(
    frame_bgr: np.ndarray,
    clip_limit: float,
    tile_grid_width: int,
    tile_grid_height: int,
) -> np.ndarray:
    """Normalise luminance while preserving colour information for inference."""

    # Custom preprocessing choice: enhance only luminance so contrast improves
    # without distorting the colour channels that the detector still uses.
    yuv_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    y_channel, u_channel, v_channel = cv2.split(yuv_frame)

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=(tile_grid_width, tile_grid_height),
    )
    enhanced_y = clahe.apply(y_channel)

    enhanced_yuv = cv2.merge((enhanced_y, u_channel, v_channel))
    return cv2.cvtColor(enhanced_yuv, cv2.COLOR_YUV2BGR)
