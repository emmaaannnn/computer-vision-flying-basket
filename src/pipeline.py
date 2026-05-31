from __future__ import annotations

import time

import cv2
import numpy as np

from src.camera import CameraStream
from src.config import VisionConfig
from src.detector import PersonDetection, PersonDetector
from src.preprocessing import apply_luminance_clahe
from src.ranging import (
    RangeCalibration,
    RangeEstimate,
    calibrate_geometric_scale,
    estimate_depth_from_height,
)


class VisionPipeline:
    """Laptop-camera validation loop for the Flying Basket vision stack."""

    def __init__(self, config: VisionConfig) -> None:
        self._config = config
        self._camera = CameraStream(
            camera_index=config.camera_index,
            frame_width=config.frame_width,
            frame_height=config.frame_height,
        )
        self._detector = PersonDetector(config)
        self._window_name = "Flying Basket Vision"

    def run(self) -> None:
        self._camera.open()

        try:
            while True:
                loop_started = time.perf_counter()
                frame = self._camera.read()
                # Custom preprocessing stage used to improve local webcam images
                # before they are passed into the pretrained detector.
                processed_frame = apply_luminance_clahe(
                    frame_bgr=frame,
                    clip_limit=self._config.clahe_clip_limit,
                    tile_grid_width=self._config.clahe_grid_width,
                    tile_grid_height=self._config.clahe_grid_height,
                )

                # Detection comes from YOLO, while range estimation and the UI
                # overlays are custom logic implemented in this repository.
                detection = self._detector.detect_primary_person(processed_frame)
                range_estimate = self._estimate_range(detection)
                calibration = self._estimate_calibration(detection)
                display_frame = self._build_display_frame(
                    raw_frame=frame,
                    processed_frame=processed_frame,
                    detection=detection,
                    range_estimate=range_estimate,
                    calibration=calibration,
                    loop_started=loop_started,
                )

                cv2.imshow(self._window_name, display_frame)

                budget_ms = 1000.0 / self._config.target_fps
                loop_ms = (time.perf_counter() - loop_started) * 1000.0
                wait_ms = max(1, int(round(max(1.0, budget_ms - loop_ms))))
                key = cv2.waitKey(wait_ms) & 0xFF

                if key in (27, ord("q")):
                    break
        finally:
            self._camera.release()
            cv2.destroyAllWindows()

    def _estimate_range(
        self,
        detection: PersonDetection | None,
    ) -> RangeEstimate | None:
        if detection is None:
            return None

        # Custom geometric ranging: use apparent person height in pixels as a
        # simple inverse-depth signal for early validation.
        return estimate_depth_from_height(
            box_height_px=detection.height_px,
            geometric_scale_px_m=self._config.geometric_scale_px_m,
            min_safe_distance_m=self._config.min_safe_distance_m,
            max_safe_distance_m=self._config.max_safe_distance_m,
        )

    def _estimate_calibration(
        self,
        detection: PersonDetection | None,
    ) -> RangeCalibration | None:
        if detection is None or self._config.reference_distance_m is None:
            return None

        # Custom one-point calibration helper for tuning the geometric scale
        # against a measured real-world distance.
        return calibrate_geometric_scale(
            box_height_px=detection.height_px,
            reference_distance_m=self._config.reference_distance_m,
            geometric_scale_px_m=self._config.geometric_scale_px_m,
        )

    def _build_display_frame(
        self,
        raw_frame: np.ndarray,
        processed_frame: np.ndarray,
        detection: PersonDetection | None,
        range_estimate: RangeEstimate | None,
        calibration: RangeCalibration | None,
        loop_started: float,
    ) -> np.ndarray:
        base_frame = processed_frame if self._config.show_preprocessed else raw_frame
        display_frame = base_frame.copy()

        if detection is not None:
            x1, y1, x2, y2 = detection.box_xyxy
            box_colour = (0, 200, 0)
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_colour, 2)
            cv2.putText(
                display_frame,
                f"person {detection.confidence:.2f}",
                (x1, max(24, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_colour,
                2,
            )

        diagnostics = self._diagnostic_lines(
            loop_started=loop_started,
            detection=detection,
            range_estimate=range_estimate,
            calibration=calibration,
        )

        for index, line in enumerate(diagnostics):
            cv2.putText(
                display_frame,
                line,
                (16, 32 + (index * 26)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

        return display_frame

    def _diagnostic_lines(
        self,
        loop_started: float,
        detection: PersonDetection | None,
        range_estimate: RangeEstimate | None,
        calibration: RangeCalibration | None,
    ) -> list[str]:
        loop_ms = (time.perf_counter() - loop_started) * 1000.0
        budget_ms = 1000.0 / self._config.target_fps

        # Custom operator-facing status text. Technical details stay optional
        # so the same loop can support both demos and debugging.
        lines = [
            self._status_line(),
            f"system speed {loop_ms:.1f} ms per frame",
        ]

        if self._config.show_technical_diagnostics:
            lines.append(f"frame budget {budget_ms:.1f} ms")

        if detection is None:
            lines.append("person not found")
            lines.append("step into view and keep your upper body visible")
            return lines

        lines.append(f"person confidence {detection.confidence:.2f}")

        if range_estimate is None:
            lines.append("distance unavailable")
            return lines

        state = "within window" if range_estimate.within_safe_window else "outside window"
        lines.append(f"estimated distance {range_estimate.raw_distance_m:.2f} m")
        lines.append(f"target range {self._safe_window_label(state)}")

        if calibration is not None:
            lines.append(f"measured distance {calibration.reference_distance_m:.2f} m")
            lines.append(f"distance error {calibration.distance_error_m:+.2f} m")
            lines.append(
                f"suggested scale {calibration.recommended_scale_px_m:.1f} px*m"
            )

        if self._config.show_technical_diagnostics:
            lines.append(
                f"box centre ({detection.x_center_px:.0f}, {detection.y_center_px:.0f}) px"
            )
            lines.append(f"box height {detection.height_px:.1f} px")
            lines.append(f"clamped distance {range_estimate.clamped_distance_m:.2f} m")

        return lines

    def _status_line(self) -> str:
        if self._config.enable_inference:
            return "person tracking ready"

        return "camera preview only"

    @staticmethod
    def _safe_window_label(state: str) -> str:
        if state == "within window":
            return "good"

        return "adjust position"
