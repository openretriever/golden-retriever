"""Real perception example: language-grounded pointing with Gemini or a mock fallback."""

from __future__ import annotations

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from examples.advanced.perception_examples.common import ColorDetector, PointPrinter, PointToLabel
from examples.advanced.real_perception.common import DEFAULT_POINT_QUERY, GeminiPointer, StaticImageSource, load_example_image


def _mock_target_from_query(query: str) -> str:
    lowered = query.lower()
    if "blue" in lowered:
        return "blue"
    return "red"


def build_pipeline(*, backend: str, image_path: str | None, query: str, dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    example = load_example_image(image_path)
    pipe = Pipeline("advanced_real_pointing")
    with pipe:
        source = StaticImageSource(image_path=image_path, dt=dt) @ Rate(hz=hz)
        if backend == "gemini_api":
            pointer = GeminiPointer(query=query) @ Trigger("image")
        else:
            detector = ColorDetector() @ Trigger("image")
            pointer = PointToLabel(
                target_label=_mock_target_from_query(query),
                image_width=example.image.shape[1],
                image_height=example.image.shape[0],
            ) @ Trigger("frame_id")
            pipe.connect(source, detector, sync=Latest())
            pipe.connect(detector, pointer, sync=Latest())
        printer = PointPrinter() @ Trigger("frame_id")
        if backend == "gemini_api":
            pipe.connect(source, pointer, sync=Latest())
        pipe.connect(pointer, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pointing flow over a static image with explicit real/mock backends.")
    parser.add_argument("--backend", choices=["mock", "gemini_api"], default="mock")
    parser.add_argument("--image", default=None)
    parser.add_argument("--query", default=DEFAULT_POINT_QUERY)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dt", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(backend=args.backend, image_path=args.image, query=args.query, dt=args.dt)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == "__main__":
    main()
