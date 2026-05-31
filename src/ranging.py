from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RangeEstimate:
    # Custom output of the repo's simple geometric ranging stage.
    """Geometric depth estimate derived from a detected target height."""

    raw_distance_m: float
    clamped_distance_m: float
    within_safe_window: bool


@dataclass(slots=True)
class RangeCalibration:
    # Custom calibration summary used to tune the scale constant from a known
    # measured distance.
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
    """Compute O(1) inverse-depth from a bounding-box height measurement."""

    if box_height_px <= 0.0:
        return None

    # Custom approximation: as the detected person appears taller in pixels,
    # the estimated camera-to-person distance becomes smaller.
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
    """Estimate a better inverse-depth scale from one measured observation."""

    if box_height_px <= 0.0 or reference_distance_m <= 0.0:
        return None

    # Custom calibration rule: derive the scale constant that would make the
    # current observation match the measured reference distance.
    predicted_distance_m = geometric_scale_px_m / box_height_px
    recommended_scale_px_m = box_height_px * reference_distance_m
    distance_error_m = predicted_distance_m - reference_distance_m

    return RangeCalibration(
        reference_distance_m=reference_distance_m,
        recommended_scale_px_m=recommended_scale_px_m,
        predicted_distance_m=predicted_distance_m,
        distance_error_m=distance_error_m,
    )
