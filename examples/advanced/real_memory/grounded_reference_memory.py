"""Real memory example: select a remembered target from explicit real/mock detections."""

from __future__ import annotations

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from examples.advanced.memory_examples.common import BeliefTracker, SelectBeliefTarget
from examples.advanced.perception_examples.common import ColorDetector, PointPrinter
from examples.advanced.real_perception.common import (
    DEFAULT_DETECTION_LABELS,
    GeminiDetector,
    StaticImageSource,
    load_example_image,
    parse_labels,
)


def build_pipeline(*, backend: str, image_path: str | None, labels: tuple[str, ...], target_label: str, dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    example = load_example_image(image_path)
    pipe = Pipeline("advanced_real_memory_reference")
    with pipe:
        source = StaticImageSource(image_path=image_path, dt=dt) @ Rate(hz=hz)
        detector = (GeminiDetector(labels=labels) if backend == "gemini_api" else ColorDetector()) @ Trigger("image")
        belief = BeliefTracker(image_width=example.image.shape[1], image_height=example.image.shape[0]) @ Trigger("frame_id")
        selector = SelectBeliefTarget(target_label=target_label) @ Trigger("frame_id")
        printer = PointPrinter() @ Trigger("frame_id")
        pipe.connect(source, detector, sync=Latest())
        pipe.connect(detector, belief, sync=Latest())
        pipe.connect(belief, selector, sync=Latest())
        pipe.connect(selector, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remember a target label through a stable belief state.")
    parser.add_argument("--backend", choices=["mock", "gemini_api"], default="mock")
    parser.add_argument("--image", default=None)
    parser.add_argument("--labels", default=",".join(DEFAULT_DETECTION_LABELS))
    parser.add_argument("--target", default="red")
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--dt", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(
        backend=args.backend,
        image_path=args.image,
        labels=parse_labels(args.labels, fallback=DEFAULT_DETECTION_LABELS),
        target_label=args.target,
        dt=args.dt,
    )
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == "__main__":
    main()
