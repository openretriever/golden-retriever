"""Minimal advanced language example: caption to primitive plan text.

Run:
  pixi run -e golden-retriever demo-language-caption-plan
  pixi run -e golden-retriever python -m examples.advanced.language_examples.caption_to_plan --steps 4 --dt 0.1
"""

from __future__ import annotations

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from examples.advanced.language_examples.common import CaptionPlanner, CaptionSource, PlanTextPrinter


def build_pipeline(*, dt: float) -> Pipeline:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline('advanced_language_caption_plan')
    with pipe:
        source = CaptionSource() @ Rate(hz=hz)
        planner = CaptionPlanner() @ Trigger('text')
        printer = PlanTextPrinter() @ Trigger('summary')
        pipe.connect(source, planner, sync=Latest())
        pipe.connect(planner, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Caption to primitive plan-text flow.')
    parser.add_argument('--steps', type=int, default=4)
    parser.add_argument('--dt', type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(dt=args.dt)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == '__main__':
    main()
