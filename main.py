"""CLI entry point for the Galen demo device detection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from detector.pipeline import DetectionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Galen demo device detection")
    parser.add_argument(
        "image",
        type=Path,
        help="Path to an input image (PNG, JPEG, etc.)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/device.yaml"),
        help="Device configuration file",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    frame = cv2.imread(str(args.image), cv2.IMREAD_UNCHANGED)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")

    pipeline = DetectionPipeline.from_config(args.config)
    result = pipeline.process(frame, frame_id=args.image.name)

    print(f"frame_id={result.metadata['frame_id']}")
    print(f"decision={result.metadata['decision']}")
    print(f"mean_intensity={result.gate.mean_intensity:.2f}")
    print(f"contrast_std={result.gate.contrast_std:.2f}")
    print(f"noise_estimate={result.gate.noise_estimate:.2f}")
    return 0 if result.gate.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
