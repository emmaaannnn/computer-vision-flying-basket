from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2

from src.tello.config import TelloFollowConfig
from src.tello.controller import TelloController
from src.vision.camera import CameraStream
from src.vision.config import VisionConfig
from src.vision.detector import PersonDetection, PersonDetector
from src.vision.ranging import estimate_depth_from_height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tello follow demo using YOLO person detector")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--model-name", type=str, default="yolo26n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--geom-scale", type=float, default=950.0)
    parser.add_argument("--target-distance", type=float, default=1.5)
    parser.add_argument("--takeoff", action="store_true")
    parser.add_argument("--no-video-window", action="store_true")
    return parser


def clamp(v: int, lo: int = -100, hi: int = 100) -> int:
    return max(lo, min(hi, int(v)))


class TelloFollower:
    """Closed-loop follow controller that combines vision with Tello commands."""

    def __init__(self, vision_config: VisionConfig, follow_config: TelloFollowConfig) -> None:
        self._vision_config = vision_config
        self._follow_config = follow_config
        self._camera = CameraStream(
            camera_index=vision_config.camera_index,
            frame_width=vision_config.frame_width,
            frame_height=vision_config.frame_height,
        )
        self._detector = PersonDetector(vision_config)
        self._tello = TelloController()

    def run(self, takeoff: bool, show_window: bool) -> int:
        self._camera.open()
        self._tello.connect()

        took_off = False
        try:
            if takeoff:
                self._tello.takeoff()
                took_off = True

            while True:
                start = time.time()
                frame = self._camera.read()
                detection: Optional[PersonDetection] = self._detector.detect_primary_person(frame)

                left_right = 0
                forward_backward = 0
                up_down = 0
                yaw = 0

                if detection is not None:
                    _, frame_width = frame.shape[:2]
                    error_x = detection.x_center_px - (frame_width / 2.0)
                    yaw = clamp(
                        int(self._follow_config.kp_yaw * error_x),
                        -self._follow_config.max_yaw,
                        self._follow_config.max_yaw,
                    )

                    range_estimate = estimate_depth_from_height(
                        box_height_px=detection.height_px,
                        geometric_scale_px_m=self._vision_config.geometric_scale_px_m,
                        min_safe_distance_m=self._vision_config.min_safe_distance_m,
                        max_safe_distance_m=self._vision_config.max_safe_distance_m,
                    )
                    if range_estimate is not None:
                        dist_error = (
                            range_estimate.raw_distance_m
                            - self._follow_config.target_distance_m
                        )
                        forward_backward = clamp(
                            int(self._follow_config.kp_forward * dist_error),
                            -self._follow_config.max_forward,
                            self._follow_config.max_forward,
                        )

                    x1, y1, x2, y2 = detection.box_xyxy
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"yaw {yaw} fwd {forward_backward}",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )

                self._tello.send_velocity(left_right, forward_backward, up_down, yaw)

                if show_window:
                    cv2.imshow("Tello Follow", frame)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                        break

                elapsed = time.time() - start
                time.sleep(max(0.0, self._follow_config.loop_dt_s - elapsed))
        finally:
            try:
                self._tello.send_velocity(0, 0, 0, 0)
            except Exception:
                pass
            if took_off:
                self._tello.land()
            self._tello.end()
            self._camera.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        return 0


def main() -> int:
    args = build_parser().parse_args()
    vision_config = VisionConfig(
        camera_index=args.camera_index,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        model_name=args.model_name,
        confidence_threshold=args.confidence,
        geometric_scale_px_m=args.geom_scale,
    )
    follow_config = TelloFollowConfig(target_distance_m=args.target_distance)
    follower = TelloFollower(vision_config, follow_config)
    return follower.run(takeoff=args.takeoff, show_window=not args.no_video_window)
