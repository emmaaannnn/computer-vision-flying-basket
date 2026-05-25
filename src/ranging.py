from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RangeEstimate:
    """Geometric depth estimate derived from a detected target height."""

    raw_distance_m: float
    clamped_distance_m: float
    within_safe_window: bool


def estimate_depth_from_height(
    box_height_px: float,
    geometric_scale_px_m: float,
    min_safe_distance_m: float,
    max_safe_distance_m: float,
) -> RangeEstimate | None:
    """Compute O(1) inverse-depth from a bounding-box height measurement."""

    if box_height_px <= 0.0:
        return None

    raw_distance_m = geometric_scale_px_m / box_height_px
    clamped_distance_m = min(
        max(raw_distance_m, min_safe_distance_m),
        max_safe_distance_m,
    )
    within_safe_window = min_safe_distance_m <= raw_distance_m <= max_safe_distance_m

    return RangeEstimate(
        raw_distance_m=raw_distance_m,
        clamped_distance_m=clamped_distance_m,
        within_safe_window=within_safe_window,
    )
