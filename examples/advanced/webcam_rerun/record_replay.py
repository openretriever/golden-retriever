"""
Record + Replay Webcam Perception Demo with MCAP.

This example demonstrates the unified MCAP recording workflow:
  1. Record: Capture webcam perception data to MCAP
  2. Visualize: Open MCAP in Rerun viewer
  3. Replay: Re-run perception pipeline from recorded data

Usage:
  # Record 50 steps to MCAP
  pixi run python examples/advanced/webcam_rerun/record_replay.py record --steps 50

  # Visualize in Rerun
  pixi run rerun webcam_session.mcap

  # Replay from MCAP (with live Rerun streaming)
  pixi run python examples/advanced/webcam_rerun/record_replay.py replay --stream
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from retriever.flow import Flow, Pipeline, Rate, Trigger, Latest, io


# =============================================================================
# Data Types
# =============================================================================


@io
@dataclass
class CameraFrame:
    """Camera frame with metadata."""

    image: np.ndarray
    frame_id: int
    timestamp: float


# =============================================================================
# Flows
# =============================================================================


class WebcamSource(Flow[None, CameraFrame]):
    """Real webcam source with mock fallback."""

    def init(self) -> None:
        self.frame_id = 0
        self.cap = None
        self._use_mock = False

        try:
            import cv2

            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                print("[Webcam] Camera not available, using mock frames")
                self._use_mock = True
                self.cap = None
            else:
                print("[Webcam] Using real camera")
        except ImportError:
            print("[Webcam] OpenCV not available, using mock frames")
            self._use_mock = True

    def cleanup(self) -> None:
        if self.cap is not None:
            self.cap.release()

    def run(self, _) -> CameraFrame:
        import time

        self.frame_id += 1
        timestamp = time.time()

        if self._use_mock or self.cap is None:
            # Generate mock gradient image
            h, w = 240, 320
            t = self.frame_id * 0.1
            x = np.linspace(0, 1, w)
            y = np.linspace(0, 1, h)
            xx, yy = np.meshgrid(x, y)

            r = ((np.sin(xx * 4 + t) + 1) * 127).astype(np.uint8)
            g = ((np.sin(yy * 4 + t) + 1) * 127).astype(np.uint8)
            b = ((np.sin((xx + yy) * 4 + t) + 1) * 127).astype(np.uint8)

            image = np.stack([r, g, b], axis=-1)
        else:
            import cv2

            ret, frame = self.cap.read()
            if ret and frame is not None:
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                # Fallback to black frame
                image = np.zeros((240, 320, 3), dtype=np.uint8)

        return CameraFrame(
            image=image,
            frame_id=self.frame_id,
            timestamp=timestamp,
        )


class PerceptionProcessor(Flow[CameraFrame, CameraFrame]):
    """Simple perception that processes the frame."""

    def run(self, frame: CameraFrame) -> CameraFrame:
        # Add some processing (e.g., brightness adjustment)
        processed = np.clip(frame.image.astype(np.int32) + 20, 0, 255).astype(np.uint8)
        return CameraFrame(
            image=processed,
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
        )


class Sink(Flow[CameraFrame, None]):
    """Minimal sink to terminate the pipeline."""

    def run(self, frame: CameraFrame) -> None:
        return None


# =============================================================================
# Pipeline Builders
# =============================================================================


def build_record_pipeline() -> tuple[Pipeline, object]:
    """Build pipeline for recording."""
    pipe = Pipeline("webcam_record")

    camera = WebcamSource() @ Rate(hz=10)
    processor = PerceptionProcessor() @ Trigger("image")
    sink = Sink() @ Rate(hz=10)

    pipe.connect(camera, processor, sync=Latest())
    pipe.connect(processor, sink, sync=Latest())

    return pipe, camera


def build_replay_pipeline() -> tuple[Pipeline, object]:
    """Build pipeline for replay."""
    pipe = Pipeline("webcam_replay")

    # Start with webcam, will be replaced with replay source
    camera = WebcamSource() @ Rate(hz=10)
    processor = PerceptionProcessor() @ Trigger("image")
    sink = Sink() @ Rate(hz=10)

    pipe.connect(camera, processor, sync=Latest())
    pipe.connect(processor, sink, sync=Latest())

    return pipe, camera


# =============================================================================
# Commands
# =============================================================================


def cmd_record(args: argparse.Namespace) -> None:
    """Record webcam perception to MCAP."""
    pipe, camera = build_record_pipeline()

    print(f"Recording {args.steps} steps to {args.output}...")
    print()

    try:
        pipe.record(
            args.output,
            steps=args.steps,
            dt=args.dt,
            visualize=args.stream,
        )
    finally:
        pipe.close_stepper()

    print()
    print(f"Done! Recording saved to: {args.output}")
    print()
    print("To visualize:")
    print(f"  pixi run python examples/advanced/webcam_rerun/record_replay.py view")


def cmd_replay(args: argparse.Namespace) -> None:
    """Replay from MCAP recording."""
    if not args.input.exists():
        print(f"Error: Recording not found: {args.input}")
        print(f"Run 'record' command first to create it.")
        return

    print(f"Replaying from {args.input}...")
    print()

    pipe, camera = build_replay_pipeline()

    # Replace camera with replay source from MCAP
    pipe.replay(camera, path=args.input)

    try:
        for i in range(args.steps):
            result = pipe.step(dt=args.dt)
            if i % 10 == 0:
                executed = ", ".join(result.executed) if result.executed else "none"
                print(f"  Step {i}: {executed}")
    finally:
        pipe.close_stepper()

    print()
    print("Replay complete!")


# =============================================================================
# CLI
# =============================================================================


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record + Replay webcam perception demo.")
    sub = p.add_subparsers(dest="cmd", required=True)

    # Record command
    record = sub.add_parser("record", help="Record webcam perception to MCAP")
    record.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("webcam_session.mcap"),
        help="Output MCAP path",
    )
    record.add_argument("--steps", type=int, default=50, help="Number of steps")
    record.add_argument("--dt", type=float, default=0.1, help="Time per step (seconds)")
    record.add_argument("--stream", action="store_true", help="Stream to Rerun live")

    # Replay command
    replay = sub.add_parser("replay", help="Replay from MCAP recording")
    replay.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("webcam_session.mcap"),
        help="Input MCAP path",
    )
    replay.add_argument("--steps", type=int, default=50, help="Max replay steps")
    replay.add_argument("--dt", type=float, default=0.1, help="Time per step (seconds)")
    replay.add_argument("--stream", action="store_true", help="Stream to Rerun live")

    # View command
    view = sub.add_parser("view", help="View MCAP recording in Rerun")
    view.add_argument(
        "--input",
        "-i",
        type=Path,
        default=Path("webcam_session.mcap"),
        help="Input MCAP path",
    )

    return p.parse_args()


def cmd_view(args: argparse.Namespace) -> None:
    """View MCAP recording in Rerun."""
    from retriever.lib.mcap import view_in_rerun

    if not args.input.exists():
        print(f"Error: Recording not found: {args.input}")
        return

    print(f"Opening {args.input} in Rerun...")
    view_in_rerun(args.input)


def main() -> None:
    args = parse_args()
    if args.cmd == "record":
        cmd_record(args)
    elif args.cmd == "replay":
        cmd_replay(args)
    elif args.cmd == "view":
        cmd_view(args)


if __name__ == "__main__":
    main()
