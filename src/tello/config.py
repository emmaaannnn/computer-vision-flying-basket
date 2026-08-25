from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class TelloFollowConfig:
    """Settings that control the drone-follow behavior."""

    target_distance_m: float = 1.5
    loop_dt_s: float = 0.15
    kp_yaw: float = 0.08
    kp_forward: float = 60.0
    max_yaw: int = 50
    max_forward: int = 60
