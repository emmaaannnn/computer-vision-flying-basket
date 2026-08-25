from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class VisionConfig:
    """Runtime configuration for the laptop-camera vision pipeline."""

    camera_index: int = 0
    frame_width: int = 1280
    frame_height: int = 720
    target_fps: float = 30.0

    clahe_clip_limit: float = 2.0
    clahe_grid_width: int = 8
    clahe_grid_height: int = 8

    model_name: str = "yolov8n.pt"
    confidence_threshold: float = 0.35

    geometric_scale_px_m: float = 950.0
    reference_distance_m: float | None = None
    min_safe_distance_m: float = 1.0
    max_safe_distance_m: float = 2.0

    enable_inference: bool = True
    show_preprocessed: bool = False
    show_technical_diagnostics: bool = False
