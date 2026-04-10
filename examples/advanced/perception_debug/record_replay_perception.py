"""
Record + replay a minimal perception pipeline.

Run:
  pixi run python examples/advanced/perception_debug/record_replay_perception.py record --out logs/perception_debug.mcap --steps 12
  pixi run python examples/advanced/perception_debug/record_replay_perception.py replay --recording logs/perception_debug.mcap --steps 12
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from retriever.flow import Pipeline, Rate, Trigger, Latest

from synthetic_color_stepper import SyntheticCamera, ColorDetector, DetectionPrinter
from replay_utils import load_synthetic_frame_buffer_from_mcap


def build_pipeline(*, dt: float) -> tuple[Pipeline, object]:
    hz = 1.0 / max(dt, 1e-6)
    pipe = Pipeline('perception_record_replay')
    with pipe:
        camera = SyntheticCamera() @ Rate(hz=hz)
        detector = ColorDetector() @ Trigger('image')
        printer = DetectionPrinter() @ Trigger('dominant_color')
        pipe.connect(camera, detector, sync=Latest())
        pipe.connect(detector, printer, sync=Latest())
    return pipe, camera


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Record/replay a synthetic perception pipeline.')
    sub = p.add_subparsers(dest='cmd', required=True)

    record = sub.add_parser('record', help='Record a short perception session to MCAP')
    record.add_argument('--out', type=Path, default=Path('logs/perception_debug.mcap'))
    record.add_argument('--steps', type=int, default=12)
    record.add_argument('--dt', type=float, default=0.1)
    record.add_argument('--stream', action='store_true', help='Stream to Rerun while recording')

    replay = sub.add_parser('replay', help='Replay an existing MCAP session')
    replay.add_argument('--recording', type=Path, default=Path('logs/perception_debug.mcap'))
    replay.add_argument('--steps', type=int, default=12)
    replay.add_argument('--dt', type=float, default=0.1)
    replay.add_argument('--sleep', type=float, default=0.0)

    return p.parse_args()


def cmd_record(args: argparse.Namespace) -> None:
    pipe, _camera = build_pipeline(dt=args.dt)
    try:
        pipe.record(args.out, steps=args.steps, dt=args.dt, visualize=args.stream)
    finally:
        pipe.close_stepper()
    print(f'[record] wrote {args.out} ({args.steps} steps)')


def cmd_replay(args: argparse.Namespace) -> None:
    if not args.recording.exists():
        raise FileNotFoundError(f'Recording not found: {args.recording}')
    pipe, camera = build_pipeline(dt=args.dt)
    camera_buffer = load_synthetic_frame_buffer_from_mcap(args.recording)
    pipe.replay(camera, buffer=camera_buffer)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
            if args.sleep > 0:
                time.sleep(args.sleep)
    finally:
        pipe.close_stepper()
    print(f'[replay] ran {args.steps} steps from {args.recording}')


def main() -> None:
    args = parse_args()
    if args.cmd == 'record':
        cmd_record(args)
        return
    if args.cmd == 'replay':
        cmd_replay(args)
        return
    raise SystemExit(f'Unknown subcommand: {args.cmd}')


if __name__ == '__main__':
    main()
