from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RangeEstimate:
    """Geometric depth estimate derived from a detected target height."""

    raw_distance_m: float
    clamped_distance_m: float
    within_safe_window: bool


@dataclass(slots=True)
class RangeCalibration:
    """Calibration guidance derived from a measured reference distance."""

    reference_distance_m: float
    recommended_scale_px_m: float
    predicted_distance_m: float
    distance_error_m: float


def estimate_depth_from_height(
    box_height_px: float,
    geometric_scale_px_m: float,
    min_safe_distance_m: float,
    max_safe_distance_m: float,
) -> RangeEstimate | None:
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


def calibrate_geometric_scale(
    box_height_px: float,
    reference_distance_m: float,
    geometric_scale_px_m: float,
) -> RangeCalibration | None:
    if box_height_px <= 0.0 or reference_distance_m <= 0.0:
        return None

    predicted_distance_m = geometric_scale_px_m / box_height_px
    recommended_scale_px_m = box_height_px * reference_distance_m
    distance_error_m = predicted_distance_m - reference_distance_m

    return RangeCalibration(
        reference_distance_m=reference_distance_m,
        recommended_scale_px_m=recommended_scale_px_m,
        predicted_distance_m=predicted_distance_m,
        distance_error_m=distance_error_m,
    )
