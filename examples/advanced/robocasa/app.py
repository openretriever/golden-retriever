"""Replay RoboCasa demonstration actions through a connected Retriever graph."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import retriever
from retriever.config import VizConfig
from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io


def _rerun_scalar(value: float) -> Any:
    import rerun as rr

    scalar = getattr(rr, "Scalar", None)
    return scalar(value) if scalar is not None else rr.Scalars([value])


@io
@dataclass
class RoboCasaAction:
    values: np.ndarray | None = None
    episode_step: int = 0
    cycle: int = 0
    active: bool = False

    def log_to_rerun(self, path: str) -> None:
        import rerun as rr

        if self.values is not None:
            rr.log(f"{path}/norm", _rerun_scalar(float(np.linalg.norm(self.values))))
        rr.log(f"{path}/episode_step", _rerun_scalar(float(self.episode_step)))
        rr.log(f"{path}/cycle", _rerun_scalar(float(self.cycle)))
        rr.log(f"{path}/active", _rerun_scalar(float(self.active)))


@io
@dataclass
class RoboCasaObservation:
    image: np.ndarray | None = None
    image_updated: bool = False
    source: str = "mock"
    task: str = "TurnOnMicrowave"
    episode_step: int = 0
    cycle: int = 0
    progress: float = 0.0
    reward: float = 0.0
    success: bool = False
    action_norm: float = 0.0

    def log_to_rerun(self, path: str) -> None:
        import rerun as rr

        if self.image is not None and self.image_updated:
            image = rr.Image(self.image)
            compress = getattr(image, "compress", None)
            rr.log(
                f"{path}/camera",
                compress(jpeg_quality=85) if callable(compress) else image,
            )
        rr.log(f"{path}/episode_step", _rerun_scalar(float(self.episode_step)))
        rr.log(f"{path}/cycle", _rerun_scalar(float(self.cycle)))
        rr.log(f"{path}/progress", _rerun_scalar(self.progress))
        rr.log(f"{path}/reward", _rerun_scalar(self.reward))
        rr.log(f"{path}/success", _rerun_scalar(float(self.success)))
        rr.log(f"{path}/action_norm", _rerun_scalar(self.action_norm))
        rr.log(f"{path}/task", rr.TextLog(self.task))


def _dataset_path(task: str, split: str) -> Path:
    from robocasa.utils.dataset_registry_utils import get_ds_meta

    metadata = get_ds_meta(task=task, split=split, source="human")
    if metadata is None:
        raise ValueError(f"No registered human dataset for {task} ({split})")
    path = Path(metadata["path"])
    if not path.exists():
        raise FileNotFoundError(
            f"No local human dataset for {task}. Download it with "
            f"`python -m robocasa.scripts.download_datasets --tasks {task} "
            f"--split {split} --source human`."
        )
    return path


class DemoActionSource(Flow[None, RoboCasaAction]):
    """Emit deterministic mock actions or a recorded human demonstration."""

    def __init__(
        self,
        *,
        mode: str = "mock",
        task: str = "TurnOnMicrowave",
        split: str = "pretrain",
        episode: int = 0,
        repeat: bool = False,
        mock_steps: int = 12,
    ) -> None:
        self.mode = mode
        self.task = task
        self.split = split
        self.episode = episode
        self.repeat = repeat
        self.mock_steps = mock_steps
        self.actions: np.ndarray | None = None
        self._index = 0
        self._cycle = 0

    def init_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task": self.task,
            "split": self.split,
            "episode": self.episode,
            "repeat": self.repeat,
            "mock_steps": self.mock_steps,
        }

    def reset(self) -> None:
        if self.mode == "robocasa":
            from robocasa.utils import lerobot_utils as lerobot

            self.actions = lerobot.get_episode_actions(
                _dataset_path(self.task, self.split), self.episode
            )
        else:
            phases = np.linspace(0.0, np.pi, self.mock_steps, dtype=np.float32)
            self.actions = np.stack(
                [np.sin(phases), np.cos(phases), np.full_like(phases, 0.25)],
                axis=1,
            )
        self._index = 0
        self._cycle = 0

    def step(self, _input: None = None) -> RoboCasaAction:
        if self.actions is None:
            raise RuntimeError("Demo actions are not initialized")
        if self._index >= len(self.actions):
            if not self.repeat:
                return RoboCasaAction()
            self._index = 0
            self._cycle += 1

        episode_step = self._index
        values = self.actions[episode_step]
        self._index += 1
        return RoboCasaAction(
            values=values,
            episode_step=episode_step,
            cycle=self._cycle,
            active=True,
        )


class RoboCasaSimulator(Flow[RoboCasaAction, RoboCasaObservation]):
    """Advance either a deterministic mock or a real RoboCasa environment."""

    _main_thread = True

    def __init__(
        self,
        *,
        mode: str = "mock",
        task: str = "TurnOnMicrowave",
        split: str = "pretrain",
        episode: int = 0,
        hz: float = 20.0,
        viewer: bool = False,
        emit_images: bool = False,
        camera: str = "robot0_agentview_center",
        width: int = 768,
        height: int = 512,
        image_hz: float = 5.0,
        mock_steps: int = 12,
    ) -> None:
        self.mode = mode
        self.task = task
        self.split = split
        self.episode = episode
        self.hz = hz
        self.viewer = viewer
        self.emit_images = emit_images
        self.camera = camera
        self.width = width
        self.height = height
        self.image_hz = image_hz
        self.mock_steps = mock_steps
        self.env: Any | None = None
        self._initial_state: dict[str, Any] | None = None
        self._action_count = mock_steps
        self._last_episode_step = -1
        self._next_step_at = 0.0
        self._frame_stride = max(1, round(hz / image_hz))
        self.latest: RoboCasaObservation | None = None

    def init_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task": self.task,
            "split": self.split,
            "episode": self.episode,
            "hz": self.hz,
            "viewer": self.viewer,
            "emit_images": self.emit_images,
            "camera": self.camera,
            "width": self.width,
            "height": self.height,
            "image_hz": self.image_hz,
            "mock_steps": self.mock_steps,
        }

    def reset(self) -> None:
        self._last_episode_step = -1
        self._next_step_at = time.monotonic()
        self.latest = None
        if self.mode == "mock" or self.env is not None:
            return

        try:
            import robocasa  # noqa: F401
            import robosuite
            from robocasa.scripts.dataset_scripts.playback_dataset import reset_to
            from robocasa.utils import lerobot_utils as lerobot
        except ImportError as exc:
            raise RuntimeError(
                "RoboCasa is not installed. Run the mock smoke first, then follow "
                "examples/advanced/robocasa/README.md for the real setup."
            ) from exc

        path = _dataset_path(self.task, self.split)
        metadata = lerobot.get_env_metadata(path)
        env_kwargs = dict(metadata["env_kwargs"])
        env_kwargs["env_name"] = metadata["env_name"]
        env_kwargs["has_renderer"] = self.viewer
        env_kwargs["has_offscreen_renderer"] = self.emit_images
        env_kwargs["use_camera_obs"] = False
        env_kwargs["renderer"] = "mjviewer"
        self.env = robosuite.make(**env_kwargs)

        states = lerobot.get_episode_states(path, self.episode)
        actions = lerobot.get_episode_actions(path, self.episode)
        self._action_count = len(actions)
        self._initial_state = {
            "states": states[0],
            "model": lerobot.get_episode_model_xml(path, self.episode),
            "ep_meta": json.dumps(lerobot.get_episode_meta(path, self.episode)),
        }
        reset_to(self.env, self._initial_state)
        print(
            f"RoboCasa {self.task} ready: Retriever connected to "
            f"{self._action_count} recorded actions."
        )

    def step(self, action: RoboCasaAction) -> RoboCasaObservation:
        if not action.active or action.values is None:
            if self.viewer and self.env is not None:
                self.env.render()
            return self.latest or RoboCasaObservation(source=self.mode, task=self.task)
        observation = (
            self._step_mock(action) if self.mode == "mock" else self._step_robocasa(action)
        )
        self.latest = observation
        return observation

    def _step_mock(self, action: RoboCasaAction) -> RoboCasaObservation:
        progress = action.episode_step / max(1, self.mock_steps - 1)
        image = None
        if self.emit_images and action.episode_step % self._frame_stride == 0:
            image = np.zeros((180, 320, 3), dtype=np.uint8)
            filled = round(progress * image.shape[1])
            image[:, :filled, 1] = 180
            image[:, filled:, 2] = 70
        return RoboCasaObservation(
            image=image,
            image_updated=image is not None,
            source="mock",
            task=self.task,
            episode_step=action.episode_step,
            cycle=action.cycle,
            progress=progress,
            reward=progress,
            success=action.episode_step >= self.mock_steps - 1,
            action_norm=float(np.linalg.norm(action.values)),
        )

    def _step_robocasa(self, action: RoboCasaAction) -> RoboCasaObservation:
        if self.env is None or self._initial_state is None:
            raise RuntimeError("RoboCasa environment is not initialized")

        from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

        if action.episode_step == 0 and self._last_episode_step >= 0:
            reset_to(self.env, self._initial_state)
        sleep_for = self._next_step_at - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        _, reward, _, _ = self.env.step(action.values)
        if self.viewer:
            self.env.render()

        image = None
        if self.emit_images and action.episode_step % self._frame_stride == 0:
            image = self.env.sim.render(
                height=self.height,
                width=self.width,
                camera_name=self.camera,
            )[::-1]
        self._last_episode_step = action.episode_step
        self._next_step_at = max(self._next_step_at + (1.0 / self.hz), time.monotonic())
        return RoboCasaObservation(
            image=image,
            image_updated=image is not None,
            source="robocasa",
            task=self.task,
            episode_step=action.episode_step,
            cycle=action.cycle,
            progress=action.episode_step / max(1, self._action_count - 1),
            reward=float(reward),
            success=bool(self.env._check_success()),
            action_norm=float(np.linalg.norm(action.values)),
        )

    def finalize(self) -> None:
        if self.env is not None:
            self.env.close()
            self.env = None
            print("Closed the Retriever-connected RoboCasa simulator.")


class ObservationPrinter(Flow[RoboCasaObservation, None]):
    def __init__(self, *, print_every: int = 4) -> None:
        self.print_every = max(1, int(print_every))
        self._last_printed: tuple[int, int] | None = None
        self._last_success = False

    def init_config(self) -> dict[str, Any]:
        return {"print_every": self.print_every}

    def reset(self) -> None:
        self._last_printed = None
        self._last_success = False

    def step(self, observation: RoboCasaObservation) -> None:
        key = (observation.cycle, observation.episode_step)
        success_transition = observation.success and not self._last_success
        should_print = observation.episode_step % self.print_every == 0 or success_transition
        if should_print and key != self._last_printed:
            print(
                f"[{observation.source} step={observation.episode_step:04d}] "
                f"progress={observation.progress:.1%} "
                f"reward={observation.reward:.3f} success={observation.success}"
            )
            self._last_printed = key
        self._last_success = observation.success


class VideoRecorder(Flow[RoboCasaObservation, None]):
    """Write emitted RGB camera frames to an MP4 artifact."""

    def __init__(self, *, path: str, fps: float) -> None:
        self.path = Path(path)
        self.fps = float(fps)
        self._writer: Any | None = None
        self._frames = 0
        self._last_frame: tuple[int, int] | None = None

    def init_config(self) -> dict[str, Any]:
        return {"path": str(self.path), "fps": self.fps}

    def reset(self) -> None:
        self._writer = None
        self._frames = 0
        self._last_frame = None

    def step(self, observation: RoboCasaObservation) -> None:
        if observation.image is None or not observation.image_updated:
            return
        frame_key = (observation.cycle, observation.episode_step)
        if frame_key == self._last_frame:
            return

        import cv2

        if self._writer is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            height, width = observation.image.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(
                str(self.path), fourcc, self.fps, (width, height)
            )
            if not self._writer.isOpened():
                raise RuntimeError(f"Could not open MP4 writer for {self.path}")

        frame_bgr = cv2.cvtColor(observation.image, cv2.COLOR_RGB2BGR)
        self._writer.write(frame_bgr)
        self._frames += 1
        self._last_frame = frame_key

    def finalize(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._frames:
            print(f"Saved {self._frames} camera frames to {self.path}.")


def build_pipeline(args: argparse.Namespace) -> tuple[Pipeline, RoboCasaSimulator]:
    video_path = getattr(args, "video", None)
    emit_images = (args.visualize == "rerun" or video_path is not None) and not args.viewer
    source = DemoActionSource(
        mode=args.mode,
        task=args.task,
        split=args.split,
        episode=args.episode,
        repeat=args.repeat,
        mock_steps=args.mock_steps,
    )
    simulator = RoboCasaSimulator(
        mode=args.mode,
        task=args.task,
        split=args.split,
        episode=args.episode,
        hz=args.hz,
        viewer=args.viewer,
        emit_images=emit_images,
        camera=args.camera,
        width=args.width,
        height=args.height,
        image_hz=args.image_hz,
        mock_steps=args.mock_steps,
    )
    printer = ObservationPrinter(print_every=args.print_every)

    pipeline = Pipeline("robocasa_demo_replay", on_lag="drop")
    with pipeline:
        actions = (source @ Rate(hz=args.hz, on_lag="drop")).named("demo_actions")
        simulation = (simulator @ Rate(hz=args.hz, on_lag="drop")).named(
            "robocasa_simulator"
        )
        output = (printer @ Trigger("episode_step")).named("observation_printer")
        actions.then(simulation, sync=Latest())
        simulation.then(output, sync=Latest())
        if video_path is not None:
            video = (
                VideoRecorder(
                    path=video_path,
                    fps=getattr(args, "video_fps", args.image_hz),
                )
                @ Trigger("episode_step")
            ).named("video_recorder")
            simulation.then(video, sync=Latest())
    return pipeline, simulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retriever-connected RoboCasa demonstration replay."
    )
    parser.add_argument("--mode", choices=["mock", "robocasa"], default="mock")
    parser.add_argument("--task", default="TurnOnMicrowave")
    parser.add_argument("--split", choices=["pretrain", "target"], default="pretrain")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--mock-steps", type=int, default=12)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--image-hz", type=float, default=5.0)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--visualize", choices=["none", "rerun"], default="none")
    parser.add_argument(
        "--rerun-mode",
        choices=["spawn", "connect", "record"],
        default="spawn",
        help="Open a viewer, connect to one, or write an .rrd artifact.",
    )
    parser.add_argument("--rerun-address", default="127.0.0.1:9876")
    parser.add_argument("--recording", default="logs/robocasa-replay.rrd")
    parser.add_argument("--video", help="Write offscreen camera frames to an MP4 file.")
    parser.add_argument("--video-fps", type=float, default=5.0)
    parser.add_argument("--repeat", action="store_true")
    parser.add_argument("--print-every", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.viewer and args.video:
        raise SystemExit(
            "--video uses the offscreen MuJoCo renderer and cannot be combined "
            "with --viewer on macOS. Run them as separate commands."
        )
    retriever.init(default_viz=VizConfig(hz=args.image_hz, fields=None))
    pipeline, simulator = build_pipeline(args)

    if args.visualize == "rerun" and args.rerun_mode == "record":
        from retriever.lib.rerun import record_session

        Path(args.recording).parent.mkdir(parents=True, exist_ok=True)
        max_steps = args.steps if args.mode == "mock" else max(1, round(args.seconds * args.hz))
        try:
            with record_session(pipeline, args.recording, auto_open=False):
                for _ in range(max_steps):
                    pipeline.step(dt=1.0 / args.hz)
                    if (
                        simulator.latest is not None
                        and simulator.latest.progress >= 1.0
                        and not args.repeat
                    ):
                        break
        finally:
            pipeline.close_stepper()
        return

    if args.mode == "mock":
        try:
            for _ in range(args.steps):
                pipeline.step(dt=1.0 / args.hz)
        finally:
            pipeline.close_stepper()
        return

    if args.viewer and args.visualize == "rerun":
        print(
            "macOS native viewer mode streams Retriever telemetry to Rerun; "
            "use headless mode for Rerun camera frames."
        )
    pipeline.run(
        backend="in-process",
        duration=args.seconds,
        visualize=None if args.visualize == "none" else args.visualize,
        blocking=True,
        backend_config={
            "rerun_config": {
                "spawn": args.rerun_mode == "spawn",
                "connect_addr": args.rerun_address,
            }
        }
        if args.visualize == "rerun"
        else None,
    )


if __name__ == "__main__":
    main()
