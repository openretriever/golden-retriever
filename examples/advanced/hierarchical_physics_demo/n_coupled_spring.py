"""
Time-aware n-coupled spring simulation (linear spring chain) with Rerun visualization.

Run:
  pixi run python examples/advanced/hierarchical_physics_demo/n_coupled_spring.py --duration 10
  pixi run python examples/advanced/hierarchical_physics_demo/n_coupled_spring.py --no-rerun --duration 2
  pixi run python examples/advanced/hierarchical_physics_demo/n_coupled_spring.py --n 7 --init-mode impulse
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from retriever.flow import Pipeline, Rate, Trigger

sys.path.append(str(Path(__file__).parent))

from flows import SimClock, NCoupledSpringSim, NCoupledSpringVizFlow, PipelineVizFlow


def build_pipeline(
    args: argparse.Namespace,
    viz_html_path: Path | None,
    viz_ascii_path: Path | None,
) -> Pipeline:
    dt = 1.0 / args.hz
    pipe = Pipeline("n_coupled_spring_rerun")

    with pipe:
        clock = SimClock(dt=dt, use_wall=args.wall_clock) @ Rate(
            hz=args.hz, on_lag=args.on_lag
        )
        sim = NCoupledSpringSim(
            n=args.n,
            mass=args.mass,
            spring_k=args.spring_k,
            rest_length=args.rest_length,
            damping=args.damping,
            gravity=args.gravity,
            init_mode=args.init_mode,
            init_amplitude=args.init_amplitude,
        ) @ Trigger("t")
        viz = NCoupledSpringVizFlow(
            trail_len=args.trail_len,
            print_every=args.print_every,
            log_rerun=not args.no_rerun,
            namespace="physics/n_coupled_spring",
        ) @ Trigger("t")

        clock.then(sim)
        sim.then(viz)

        if viz_html_path is not None:
            html = PipelineVizFlow(
                str(viz_html_path),
                ascii_path=str(viz_ascii_path) if viz_ascii_path else None,
                log_rerun=not args.no_rerun,
                namespace="physics/n_coupled_spring",
            ) @ Trigger("t")
            clock.then(html)

    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Time-aware n-coupled spring with Rerun visualization."
    )
    p.add_argument(
        "--backend", default="dora", choices=["multiprocessing", "dora", "in-process"]
    )
    p.add_argument("--duration", type=float, default=10.0)
    p.add_argument("--hz", type=float, default=120.0)
    p.add_argument("--trail-len", type=int, default=200)
    p.add_argument("--n", type=int, default=5, help="Number of coupled masses")
    p.add_argument("--mass", type=float, default=1.0, help="Mass of each object (kg)")
    p.add_argument("--spring-k", type=float, default=10.0, help="Spring constant (N/m)")
    p.add_argument("--rest-length", type=float, default=1.0, help="Spring rest length (m)")
    p.add_argument("--damping", type=float, default=0.02, help="Damping coefficient")
    p.add_argument("--gravity", type=float, default=0.0, help="Gravitational acceleration (m/s^2)")
    p.add_argument(
        "--init-mode",
        default="wave",
        choices=["wave", "impulse", "compress", "random"],
        help="Initial condition mode",
    )
    p.add_argument(
        "--init-amplitude", type=float, default=0.5, help="Initial displacement amplitude (m)"
    )
    p.add_argument(
        "--on-lag", default="catch_up", choices=["warn", "drop", "catch_up", "error"]
    )
    p.add_argument(
        "--wall-clock",
        action="store_true",
        help="Use wall-clock dt instead of fixed dt.",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=60,
        help="Print progress every N steps (0 disables).",
    )
    p.add_argument(
        "--no-rerun", action="store_true", help="Disable Rerun visualization."
    )
    p.add_argument(
        "--no-viz-html",
        action="store_true",
        help="Disable the pipeline visualization HTML output.",
    )
    p.add_argument(
        "--no-open-viz",
        action="store_true",
        help="Disable auto-opening the pipeline visualization HTML.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    want_viz_html = not args.no_viz_html
    want_open_viz = not args.no_open_viz
    viz_html_path = None
    viz_ascii_path = None
    if want_viz_html:
        viz_html_path = Path("viz-n_coupled_spring_rerun-pipeline.html").resolve()
        viz_ascii_path = Path("viz-n_coupled_spring_rerun-pipeline.txt").resolve()

    pipe = build_pipeline(
        args,
        viz_html_path if (want_viz_html and not args.no_rerun) else None,
        viz_ascii_path if (want_viz_html and not args.no_rerun) else None,
    )

    print(
        f"[n_coupled_spring] backend={args.backend} duration={args.duration}s "
        f"n={args.n} spring_k={args.spring_k} init_mode={args.init_mode} "
        f"rerun={'off' if args.no_rerun else 'on'} "
        f"pipeline_viz={'off' if not want_viz_html else 'on'}"
    )

    visualize = None if args.no_rerun else "rerun"
    if viz_html_path is not None:
        ir = pipe.validate()
        ir.visualize(viz_html_path, open_browser=want_open_viz)
        if viz_ascii_path is not None:
            viz_ascii_path.write_text(ir.to_ascii())
    pipe.run(
        backend=args.backend, duration=args.duration, visualize=visualize, blocking=True
    )


if __name__ == "__main__":
    main()
