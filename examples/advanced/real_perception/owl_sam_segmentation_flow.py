"""Real perception example: segmentation with a local OWLv2+SAM backend or a mock fallback."""

from __future__ import annotations

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from examples.advanced.perception_examples.common import ColorSegmenter, SegmentationPrinter
from examples.advanced.real_perception.common import DEFAULT_SEGMENT_LABELS, OwlSamSegmenter, StaticImageSource, parse_labels


def build_pipeline(*, backend: str, image_path: str | None, labels: tuple[str, ...], dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline("advanced_real_segmentation")
    with pipe:
        source = StaticImageSource(image_path=image_path, dt=dt) @ Rate(hz=hz)
        segmenter = (OwlSamSegmenter(labels=labels) if backend == "owl_sam_local" else ColorSegmenter()) @ Trigger("image")
        printer = SegmentationPrinter() @ Trigger("frame_id")
        pipe.connect(source, segmenter, sync=Latest())
        pipe.connect(segmenter, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Segmentation flow over a static image with explicit real/mock backends.")
    parser.add_argument("--backend", choices=["mock", "owl_sam_local"], default="mock")
    parser.add_argument("--image", default=None)
    parser.add_argument("--labels", default=",".join(DEFAULT_SEGMENT_LABELS))
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--dt", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(
        backend=args.backend,
        image_path=args.image,
        labels=parse_labels(args.labels, fallback=DEFAULT_SEGMENT_LABELS),
        dt=args.dt,
    )
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == "__main__":
    main()
