from __future__ import annotations

import argparse
import time

import cv2

from src.vision.detector import PersonDetector
from src.tello.video_stream import TelloVideoStream


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show Tello video on the laptop and run YOLO26n on it.")
    parser.add_argument("--model-name", type=str, default="./yolo26n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--show-technical-diagnostics", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    detector = PersonDetector(
        model_name=args.model_name,
        confidence_threshold=args.confidence,
    )
    stream = TelloVideoStream()

    stream.open()
    try:
        while True:
            loop_started = time.perf_counter()
            frame = stream.read()
            detection = detector.detect_primary_person(frame)

            if detection is not None:
                x1, y1, x2, y2 = detection.box_xyxy
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"person {detection.confidence:.2f}",
                    (x1, max(24, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                if args.show_technical_diagnostics:
                    cv2.putText(
                        frame,
                        f"center ({detection.x_center_px:.0f}, {detection.y_center_px:.0f})",
                        (16, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"height {detection.height_px:.1f} px",
                        (16, 58),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2,
                    )
            else:
                cv2.putText(
                    frame,
                    "person not found",
                    (16, 32),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            elapsed_ms = (time.perf_counter() - loop_started) * 1000.0
            cv2.putText(
                frame,
                f"loop {elapsed_ms:.1f} ms",
                (16, frame.shape[0] - 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Tello YOLO", frame)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        stream.release()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
