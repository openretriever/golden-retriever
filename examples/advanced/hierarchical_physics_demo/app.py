"""
Hierarchical physics demo with two time-aware simulations.

Run:
  pixi run python examples/advanced/hierarchical_physics_demo/app.py --demo double_pendulum --duration 6
  pixi run python examples/advanced/hierarchical_physics_demo/app.py --demo three_body --duration 8
  pixi run python examples/advanced/hierarchical_physics_demo/app.py --demo both --duration 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retriever.flow import Pipeline, Rate, Trigger

sys.path.append(str(Path(__file__).parent))

from flows import (
    SimClock,
    DoublePendulumSim,
    DoublePendulumVizFlow,
    NBodySim,
    NBodyVizFlow,
    PipelineVizFlow,
)


def add_double_pendulum(pipe: Pipeline, args: argparse.Namespace):
    dt = 1.0 / args.pendulum_hz
    clock = SimClock(dt=dt, use_wall=args.wall_clock) @ Rate(hz=args.pendulum_hz, on_lag=args.on_lag)
    sim = DoublePendulumSim(damping=args.damping) @ Trigger("t")
    viz = DoublePendulumVizFlow(
        trail_len=args.pendulum_trail_len,
        print_every=args.print_every,
        log_rerun=not args.no_rerun,
        namespace="physics/double_pendulum",
    ) @ Trigger("t")

    clock.then(sim)
    sim.then(viz)
    return clock


def add_three_body(pipe: Pipeline, args: argparse.Namespace):
    dt = 1.0 / args.three_body_hz
    clock = SimClock(dt=dt, use_wall=args.wall_clock) @ Rate(hz=args.three_body_hz, on_lag=args.on_lag)
    sim = NBodySim(gravity=args.gravity) @ Trigger("t")
    viz = NBodyVizFlow(
        trail_len=args.three_body_trail_len,
        print_every=args.print_every,
        log_rerun=not args.no_rerun,
        namespace="physics/three_body",
    ) @ Trigger("t")

    clock.then(sim)
    sim.then(viz)
    return clock


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hierarchical physics demo (double pendulum + three-body).")
    p.add_argument("--demo", default="three_body", choices=["double_pendulum", "three_body", "both"])
    p.add_argument("--backend", default="dora", choices=["multiprocessing", "dora", "in-process"])
    p.add_argument("--duration", type=float, default=8.0)
    p.add_argument("--print-every", type=int, default=50, help="Print every N steps (0 disables).")
    p.add_argument("--on-lag", default="catch_up", choices=["warn", "drop", "catch_up", "error"])
    p.add_argument("--wall-clock", action="store_true", help="Use wall-clock dt instead of fixed dt.")
    p.add_argument("--no-rerun", action="store_true", help="Disable Rerun visualization.")
    p.add_argument("--no-viz-html", action="store_true", help="Disable the pipeline visualization HTML output.")
    p.add_argument("--no-open-viz", action="store_true", help="Disable auto-opening the pipeline visualization HTML.")

    p.add_argument("--pendulum-hz", type=float, default=120.0)
    p.add_argument("--pendulum-trail-len", type=int, default=200)
    p.add_argument("--damping", type=float, default=0.01)

    p.add_argument("--three-body-hz", type=float, default=200.0)
    p.add_argument("--three-body-trail-len", type=int, default=400)
    p.add_argument("--gravity", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pipe = Pipeline("hierarchical_physics_demo")
    want_viz_html = not args.no_viz_html
    want_open_viz = not args.no_open_viz
    viz_html_path = None
    viz_ascii_path = None
    if want_viz_html:
        viz_html_path = Path("viz-hierarchical_physics_demo-pipeline.html").resolve()
        viz_ascii_path = Path("viz-hierarchical_physics_demo-pipeline.txt").resolve()

    with pipe:
        clock_handle = None
        if args.demo in {"double_pendulum", "both"}:
            clock_handle = add_double_pendulum(pipe, args)
        if args.demo in {"three_body", "both"}:
            clock_handle = clock_handle or add_three_body(pipe, args)

        if (not args.no_rerun) and want_viz_html and viz_html_path is not None and clock_handle is not None:
            html = PipelineVizFlow(
                str(viz_html_path),
                ascii_path=str(viz_ascii_path) if viz_ascii_path else None,
                log_rerun=not args.no_rerun,
                namespace="physics/pipeline",
            ) @ Trigger("t")
            clock_handle.then(html)

    print(
        f"[hierarchical_physics_demo] demo={args.demo} backend={args.backend} "
        f"duration={args.duration}s rerun={'off' if args.no_rerun else 'on'} "
        f"pipeline_viz={'off' if not want_viz_html else 'on'}"
    )

    visualize = None if args.no_rerun else "rerun"
    if viz_html_path is not None:
        ir = pipe.validate()
        ir.visualize(viz_html_path, open_browser=want_open_viz)
        if viz_ascii_path is not None:
            viz_ascii_path.write_text(ir.to_ascii())
    pipe.run(backend=args.backend, duration=args.duration, visualize=visualize, blocking=True)


if __name__ == "__main__":
    main()
