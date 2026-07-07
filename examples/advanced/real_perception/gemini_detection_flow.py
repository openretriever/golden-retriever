"""Real perception example: detection with a Gemini or mock backend.

Run:
  pixi run -e golden-retriever-perception demo-gemini-detection-flow
  pixi run -e golden-retriever-perception python -m examples.advanced.real_perception.gemini_detection_flow --backend gemini_api --labels "red block,blue block"
"""

from __future__ import annotations

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from examples.advanced.perception_examples.common import ColorDetector, DetectionPrinter
from examples.advanced.real_perception.common import DEFAULT_DETECTION_LABELS, GeminiDetector, StaticImageSource, parse_labels


def build_pipeline(*, backend: str, image_path: str | None, labels: tuple[str, ...], dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline("advanced_real_detection")
    with pipe:
        source = StaticImageSource(image_path=image_path, dt=dt) @ Rate(hz=hz)
        detector = (GeminiDetector(labels=labels) if backend == "gemini_api" else ColorDetector()) @ Trigger("image")
        printer = DetectionPrinter() @ Trigger("frame_id")
        pipe.connect(source, detector, sync=Latest())
        pipe.connect(detector, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detection flow over a static image with explicit real/mock backends.")
    parser.add_argument("--backend", choices=["mock", "gemini_api"], default="mock")
    parser.add_argument("--image", default=None, help="Optional image path. If omitted, a synthetic red/blue scene is rendered.")
    parser.add_argument("--labels", default=",".join(DEFAULT_DETECTION_LABELS))
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(
        backend=args.backend,
        image_path=args.image,
        labels=parse_labels(args.labels, fallback=DEFAULT_DETECTION_LABELS),
        dt=args.dt,
    )
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == "__main__":
    main()
