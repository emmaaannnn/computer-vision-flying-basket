from __future__ import annotations

import argparse

from src.vision.config import VisionConfig
from src.vision.pipeline import VisionPipeline


def build_parser() -> argparse.ArgumentParser:
    """Construct the command-line interface for local webcam validation."""

    parser = argparse.ArgumentParser(
        description="Run the Flying Basket laptop-camera vision pipeline."
    )
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--frame-width", type=int, default=1280)
    parser.add_argument("--frame-height", type=int, default=720)
    parser.add_argument("--target-fps", type=float, default=30.0)
    parser.add_argument("--model-name", type=str, default="yolov8n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--geom-scale", type=float, default=950.0)
    parser.add_argument("--reference-distance-m", type=float, default=None)
    parser.add_argument("--safe-min", type=float, default=1.0)
    parser.add_argument("--safe-max", type=float, default=2.0)
    parser.add_argument("--disable-inference", action="store_true")
    parser.add_argument("--show-preprocessed", action="store_true")
    parser.add_argument("--show-technical-diagnostics", action="store_true")
    return parser


def main() -> int:
    """Run the local vision loop until the user exits the display window."""

    args = build_parser().parse_args()
    config = VisionConfig(
        camera_index=args.camera_index,
        frame_width=args.frame_width,
        frame_height=args.frame_height,
        target_fps=args.target_fps,
        model_name=args.model_name,
        confidence_threshold=args.confidence,
        geometric_scale_px_m=args.geom_scale,
        reference_distance_m=args.reference_distance_m,
        min_safe_distance_m=args.safe_min,
        max_safe_distance_m=args.safe_max,
        enable_inference=not args.disable_inference,
        show_preprocessed=args.show_preprocessed,
        show_technical_diagnostics=args.show_technical_diagnostics,
    )

    pipeline = VisionPipeline(config)
    pipeline.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
