from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from src.tello.video_stream import TelloVideoStream
from src.vision.detector import PersonDetection, PersonDetector

DETECTOR_CHOICES = ("opencv", "yolo11n", "yolo26n")
DEFAULT_MODEL_PATHS = {
    "yolo11n": "./yolo11n.pt",
    "yolo26n": "./yolo26n.pt",
}


@dataclass(slots=True)
class FollowCommand:
    left_right: int = 0
    forward_backward: int = 0
    up_down: int = 0
    yaw: int = 0


class OpenCVHumanDetector:
    """OpenCV HOG-based human detector for a lightweight laptop baseline."""

    def __init__(self) -> None:
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect_primary_person(self, frame_bgr: np.ndarray) -> PersonDetection | None:
        rects, weights = self._hog.detectMultiScale(
            frame_bgr,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )

        if len(rects) == 0:
            return None

        best_index = max(range(len(rects)), key=lambda index: rects[index][2] * rects[index][3])
        x, y, width, height = rects[best_index]
        confidence_values = np.asarray(weights).reshape(-1)
        confidence = float(confidence_values[best_index]) if best_index < len(confidence_values) else 0.0

        return PersonDetection(
            x_center_px=float(x + width / 2.0),
            y_center_px=float(y + height / 2.0),
            width_px=float(width),
            height_px=float(height),
            confidence=confidence,
            box_xyxy=(int(x), int(y), int(x + width), int(y + height)),
        )


def build_parser(default_detector: str = "opencv") -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Follow a human with DJI Tello using OpenCV or YOLO.")
    parser.add_argument("--detector", choices=DETECTOR_CHOICES, default=default_detector)
    parser.add_argument("--model-path", "--model-name", dest="model_path", type=str, default=None)
    parser.add_argument("--distance-cm", "--target-distance-cm", dest="distance_cm", type=int, default=120)
    parser.add_argument("--height-cm", "--target-height-cm", dest="height_cm", type=int, default=120)
    parser.add_argument("--takeoff", action="store_true")
    parser.add_argument("--log-file", type=str, default="logs/tello_follow.csv")
    parser.add_argument("--show-technical-diagnostics", action="store_true")
    return parser


def _target_person_height_px(distance_cm: int) -> float:
    baseline_distance_cm = 120.0
    baseline_height_px = 220.0
    safe_distance_cm = max(40, distance_cm)
    return baseline_height_px * (baseline_distance_cm / float(safe_distance_cm))


def _clamp_speed(value: float, max_speed: int) -> int:
    return int(max(-max_speed, min(max_speed, round(value))))


def _gain_control(error_value: float, deadband: float, gain: float, max_speed: int) -> int:
    if abs(error_value) <= deadband:
        return 0
    return _clamp_speed(error_value * gain, max_speed)


def _choose_detector(detector_name: str, model_path: str | None, confidence: float):
    if detector_name == "opencv":
        return OpenCVHumanDetector()

    resolved_model_path = model_path or DEFAULT_MODEL_PATHS[detector_name]
    return PersonDetector(model_name=resolved_model_path, confidence_threshold=confidence)


def _detect_person(detector, frame_bgr: np.ndarray) -> PersonDetection | None:
    return detector.detect_primary_person(frame_bgr)


def _draw_text(frame: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)


def _log_row(writer: csv.DictWriter, row: dict[str, object]) -> None:
    writer.writerow(row)


def main(default_detector: str = "opencv") -> int:
    args = build_parser(default_detector=default_detector).parse_args()
    confidence = 0.35
    yaw_gain = 0.18
    forward_gain = 0.18
    altitude_gain = 0.55
    deadband_px = 35.0
    deadband_cm = 10.0
    max_speed = 25

    detector = _choose_detector(args.detector, args.model_path, confidence)
    stream = TelloVideoStream()

    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "timestamp_s",
        "detector",
        "model_path",
        "frame_width_px",
        "frame_height_px",
        "detected",
        "confidence",
        "box_x1",
        "box_y1",
        "box_x2",
        "box_y2",
        "target_height_cm",
        "target_distance_cm",
        "current_height_cm",
        "x_error_px",
        "y_error_px",
        "height_error_px",
        "altitude_error_cm",
        "left_right",
        "forward_backward",
        "up_down",
        "yaw",
    ]

    stream.open()
    try:
        if args.takeoff:
            stream.takeoff()
            time.sleep(2.0)

        with log_path.open("a", newline="") as log_file:
            writer = csv.DictWriter(log_file, fieldnames=fieldnames)
            if log_file.tell() == 0:
                writer.writeheader()

            while True:
                loop_started = time.perf_counter()
                frame = stream.read()
                detection = _detect_person(detector, frame)
                current_height_cm = stream.get_height_cm()
                target_person_height_px = _target_person_height_px(args.distance_cm)

                frame_center_x = frame.shape[1] / 2.0
                frame_center_y = frame.shape[0] / 2.0
                command = FollowCommand()
                x_error_px = 0.0
                y_error_px = 0.0
                height_error_px = 0.0
                altitude_error_cm = 0.0

                if detection is not None:
                    x1, y1, x2, y2 = detection.box_xyxy
                    x_error_px = detection.x_center_px - frame_center_x
                    y_error_px = detection.y_center_px - frame_center_y
                    height_error_px = target_person_height_px - detection.height_px
                    altitude_error_cm = float(args.height_cm - current_height_cm) if current_height_cm is not None else 0.0

                    command = FollowCommand(
                        left_right=0,
                        forward_backward=_gain_control(height_error_px, deadband_px, forward_gain, max_speed),
                        up_down=_gain_control(altitude_error_cm, deadband_cm, altitude_gain, max_speed),
                        yaw=_gain_control(x_error_px, deadband_px, yaw_gain, max_speed),
                    )

                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    _draw_text(frame, f"{args.detector} {detection.confidence:.2f}", x1, max(24, y1 - 10), (0, 255, 0))
                else:
                    _draw_text(frame, f"{args.detector} no person found", 16, 32, (255, 255, 255))

                if args.show_technical_diagnostics:
                    _draw_text(frame, f"height {current_height_cm if current_height_cm is not None else 'n/a'} cm", 16, 58, (255, 255, 255))
                    _draw_text(frame, f"target height {args.height_cm} cm", 16, 84, (255, 255, 255))
                    _draw_text(frame, f"target distance {args.distance_cm} cm", 16, 110, (255, 255, 255))
                    _draw_text(frame, f"frame center ({frame_center_x:.0f}, {frame_center_y:.0f})", 16, 136, (255, 255, 255))
                    _draw_text(frame, f"x error {x_error_px:.0f} px | h error {height_error_px:.0f} px", 16, 162, (255, 255, 255))

                elapsed_ms = (time.perf_counter() - loop_started) * 1000.0
                _draw_text(frame, f"loop {elapsed_ms:.1f} ms", 16, frame.shape[0] - 16, (255, 255, 255))

                if args.takeoff:
                    stream.send_velocity(command.left_right, command.forward_backward, command.up_down, command.yaw)

                cv2.imshow("Tello Follow", frame)

                _log_row(
                    writer,
                    {
                        "timestamp_s": f"{time.time():.3f}",
                        "detector": args.detector,
                        "model_path": args.model_path or "",
                        "frame_width_px": frame.shape[1],
                        "frame_height_px": frame.shape[0],
                        "detected": detection is not None,
                        "confidence": f"{detection.confidence:.3f}" if detection is not None else "",
                        "box_x1": detection.box_xyxy[0] if detection is not None else "",
                        "box_y1": detection.box_xyxy[1] if detection is not None else "",
                        "box_x2": detection.box_xyxy[2] if detection is not None else "",
                        "box_y2": detection.box_xyxy[3] if detection is not None else "",
                        "target_height_cm": args.height_cm,
                        "target_distance_cm": args.distance_cm,
                        "current_height_cm": current_height_cm if current_height_cm is not None else "",
                        "x_error_px": f"{x_error_px:.2f}",
                        "y_error_px": f"{y_error_px:.2f}",
                        "height_error_px": f"{height_error_px:.2f}",
                        "altitude_error_cm": f"{altitude_error_cm:.2f}",
                        "left_right": command.left_right,
                        "forward_backward": command.forward_backward,
                        "up_down": command.up_down,
                        "yaw": command.yaw,
                    },
                )
                log_file.flush()

                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
    finally:
        stream.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())