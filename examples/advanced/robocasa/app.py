"""Replay RoboCasa demonstrations through a connected Retriever graph."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import retriever
from retriever.config import VizConfig
from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io

from .embodied import (
    EmbodiedGoal,
    EmbodiedPlannerFlow,
    ExecutionState,
    GoalSource,
    SkillDispatcher,
    create_planner,
    materialize_skill_plan,
)
from .mjviser_bridge import MjviserBridge, ReplayControls
from .web_console import RetrieverWebConsole


def _rerun_scalar(value: float) -> Any:
    import rerun as rr

    scalar = getattr(rr, "Scalar", None)
    return scalar(value) if scalar is not None else rr.Scalars([value])


@io
@dataclass
class RoboCasaAction:
    values: np.ndarray | None = None
    recorded_state: np.ndarray | None = None
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
            f"`pixi run --locked -e robocasa python -m "
            f"robocasa.scripts.download_datasets --tasks {task} --split {split} "
            f"--source human`."
        )
    return path


def _reset_to_episode(env: Any, initial_state: dict[str, Any]) -> None:
    from robocasa.scripts.dataset_scripts.playback_dataset import reset_to

    reset_to(env, initial_state)


class DemoActionSource(Flow[ExecutionState, RoboCasaAction]):
    """Emit a recorded demonstration selected by a validated skill plan."""

    def __init__(
        self,
        *,
        mode: str = "mock",
        task: str = "TurnOnMicrowave",
        split: str = "pretrain",
        episode: int = 0,
        repeat: bool = False,
        mock_steps: int = 12,
        controls: ReplayControls | None = None,
    ) -> None:
        self.mode = mode
        self.task = task
        self.split = split
        self.episode = episode
        self.repeat = repeat
        self.mock_steps = mock_steps
        self.controls = controls
        self.actions: np.ndarray | None = None
        self.states: np.ndarray | None = None
        self._index = 0
        self._cycle = 0
        self._completed_cycle: int | None = None

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

            path = _dataset_path(self.task, self.split)
            self.actions = lerobot.get_episode_actions(path, self.episode)
            self.states = lerobot.get_episode_states(path, self.episode)
            if self.states is not None and len(self.states) != len(self.actions):
                raise ValueError(
                    "RoboCasa demonstration states and actions must have matching "
                    f"lengths, got {len(self.states)} states and "
                    f"{len(self.actions)} actions"
                )
        else:
            phases = np.linspace(0.0, np.pi, self.mock_steps, dtype=np.float32)
            self.actions = np.stack(
                [np.sin(phases), np.cos(phases), np.full_like(phases, 0.25)],
                axis=1,
            )
            self.states = None
        self._index = 0
        self._cycle = 0
        self._completed_cycle = None
        if self.controls is not None:
            self.controls.set_total_steps(len(self.actions))

    def step(self, execution: ExecutionState | None = None) -> RoboCasaAction:
        if self.actions is None:
            raise RuntimeError("Demo actions are not initialized")
        if execution is not None:
            planned_goal = materialize_skill_plan(execution.plan).goal
            if (planned_goal.task, planned_goal.episode) != (self.task, self.episode):
                raise ValueError(
                    "The dispatched plan does not match the loaded demonstration: "
                    f"plan={planned_goal.task} episode {planned_goal.episode}, "
                    f"replay={self.task} episode {self.episode}. Switch tasks through "
                    "the launcher so the dataset and simulator are replaced together."
                )
        if self.controls is not None:
            advance, restart_cycle = self.controls.claim_next_action()
            if restart_cycle is not None:
                self._index = 0
                self._cycle = restart_cycle
                self._completed_cycle = None
            if not advance:
                if self._index >= len(self.actions) and not self.repeat:
                    self._mark_complete_once()
                return RoboCasaAction(
                    episode_step=max(0, self._index - 1),
                    cycle=self._cycle,
                )
        if self._index >= len(self.actions):
            if not self.repeat:
                self._mark_complete_once()
                return RoboCasaAction(
                    episode_step=max(0, len(self.actions) - 1),
                    cycle=self._cycle,
                )
            self._mark_complete_once()
            self._index = 0
            self._cycle = (
                self.controls.begin_repeat_cycle()
                if self.controls is not None
                else self._cycle + 1
            )
            self._completed_cycle = None

        episode_step = self._index
        values = self.actions[episode_step]
        self._index += 1
        return RoboCasaAction(
            values=values,
            recorded_state=(
                self.states[episode_step] if self.states is not None else None
            ),
            episode_step=episode_step,
            cycle=self._cycle,
            active=True,
        )

    def _mark_complete_once(self) -> None:
        if self.controls is None or self._completed_cycle == self._cycle:
            return
        self.controls.mark_complete()
        self._completed_cycle = self._cycle


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
        visualize: str = "none",
        viser_host: str = "127.0.0.1",
        viser_port: int = 8085,
        console_host: str = "127.0.0.1",
        console_port: int = 8086,
        native_viser_controls: bool = False,
        emit_images: bool = False,
        camera: str = "robot0_agentview_center",
        width: int = 768,
        height: int = 512,
        image_hz: float = 5.0,
        mock_steps: int = 12,
        controls: ReplayControls | None = None,
        open_browser: bool = False,
        launch_id: str = "",
    ) -> None:
        self.mode = mode
        self.task = task
        self.split = split
        self.episode = episode
        self.hz = hz
        self.viewer = viewer
        self.visualize = visualize
        self.viser_host = viser_host
        self.viser_port = viser_port
        self.console_host = console_host
        self.console_port = console_port
        self.native_viser_controls = native_viser_controls
        self.emit_images = emit_images
        self.camera = camera
        self.width = width
        self.height = height
        self.image_hz = image_hz
        self.mock_steps = mock_steps
        self.controls = controls
        self.env: Any | None = None
        self._initial_state: dict[str, Any] | None = None
        self._action_count = mock_steps
        self._last_episode_step = -1
        self._active_cycle: int | None = None
        self._next_step_at = 0.0
        self._frame_stride = max(1, round(hz / image_hz))
        self.latest: RoboCasaObservation | None = None
        self._web_viewer = (
            MjviserBridge(
                host=viser_host,
                port=viser_port,
                label=f"Retriever RoboCasa {task}",
                controls=controls if native_viser_controls else None,
                open_browser=False,
            )
            if visualize == "mjviser"
            else None
        )
        display_host = (
            "localhost" if viser_host in {"0.0.0.0", "127.0.0.1"} else viser_host
        )
        self._web_console = (
            RetrieverWebConsole(
                controls,
                f"http://{display_host}:{viser_port}",
                host=console_host,
                port=console_port,
                camera_handler=self._web_viewer.apply_camera_preset,
                open_browser=open_browser,
                launch_id=launch_id,
            )
            if self._web_viewer is not None and controls is not None
            else None
        )

    def init_config(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "task": self.task,
            "split": self.split,
            "episode": self.episode,
            "hz": self.hz,
            "viewer": self.viewer,
            "visualize": self.visualize,
            "viser_host": self.viser_host,
            "viser_port": self.viser_port,
            "console_host": self.console_host,
            "console_port": self.console_port,
            "native_viser_controls": self.native_viser_controls,
            "emit_images": self.emit_images,
            "camera": self.camera,
            "width": self.width,
            "height": self.height,
            "image_hz": self.image_hz,
            "mock_steps": self.mock_steps,
        }

    def reset(self) -> None:
        self._last_episode_step = -1
        self._active_cycle = None
        self._next_step_at = time.monotonic()
        self.latest = None
        if self.mode == "mock":
            return
        if self.env is not None:
            if self._initial_state is None:
                raise RuntimeError("RoboCasa initial episode state is unavailable")
            _reset_to_episode(self.env, self._initial_state)
            if self._web_viewer is not None:
                self._web_viewer.update(self.env.sim)
            return

        try:
            import robocasa  # noqa: F401
            import robosuite
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
        env = robosuite.make(**env_kwargs)
        try:
            states = lerobot.get_episode_states(path, self.episode)
            actions = lerobot.get_episode_actions(path, self.episode)
            action_count = len(actions)
            initial_state = {
                "states": states[0],
                "model": lerobot.get_episode_model_xml(path, self.episode),
                "ep_meta": json.dumps(lerobot.get_episode_meta(path, self.episode)),
            }
            _reset_to_episode(env, initial_state)
            self.env = env
            self._action_count = action_count
            self._initial_state = initial_state
            if self.controls is not None:
                self.controls.set_total_steps(self._action_count)
            if self._web_viewer is not None:
                self._web_viewer.update(self.env.sim)
            if self._web_console is not None:
                self._web_console.start()
        except BaseException:
            self.env = None
            self._initial_state = None
            if self._web_console is not None:
                with suppress(Exception):
                    self._web_console.close()
            if self._web_viewer is not None:
                with suppress(Exception):
                    self._web_viewer.close()
            with suppress(Exception):
                env.close()
            raise
        print(
            f"RoboCasa {self.task} ready: Retriever connected to "
            f"{self._action_count} recorded actions."
        )

    def step(self, action: RoboCasaAction) -> RoboCasaObservation:
        if not action.active or (
            action.values is None and action.recorded_state is None
        ):
            if self.viewer and self.env is not None:
                self.env.render()
            if self._web_viewer is not None:
                self._web_viewer.refresh_controls()
            return self.latest or RoboCasaObservation(source=self.mode, task=self.task)
        observation = (
            self._step_mock(action)
            if self.mode == "mock"
            else self._step_robocasa(action)
        )
        self.latest = observation
        return observation

    def refresh_controls(self) -> None:
        """Refresh renderer controls after a complete Retriever pipeline tick."""

        if self._web_viewer is not None:
            self._web_viewer.refresh_controls()

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

        if self._active_cycle is not None and action.cycle < self._active_cycle:
            return self.latest or RoboCasaObservation(source=self.mode, task=self.task)
        cycle_changed = self._active_cycle is None or action.cycle != self._active_cycle
        if self._active_cycle is not None and cycle_changed:
            _reset_to_episode(self.env, self._initial_state)
            self._last_episode_step = -1
            self._next_step_at = time.monotonic()
        self._active_cycle = action.cycle
        if action.episode_step <= self._last_episode_step:
            return self.latest or RoboCasaObservation(source=self.mode, task=self.task)
        expected_step = self._last_episode_step + 1
        if action.recorded_state is None and action.episode_step != expected_step:
            raise RuntimeError(
                "Open-loop RoboCasa action replay skipped a recorded action: "
                f"expected step {expected_step}, received {action.episode_step}. "
                "Use demonstration mode for deterministic state playback."
            )
        sleep_for = self._next_step_at - time.monotonic()
        if sleep_for > 0:
            time.sleep(sleep_for)
        if action.recorded_state is not None:
            _reset_to_episode(self.env, {"states": action.recorded_state})
            success = bool(self.env._check_success())
            reward_fn = getattr(self.env, "reward", None)
            reward = float(reward_fn()) if callable(reward_fn) else float(success)
        else:
            if action.values is None:
                raise RuntimeError("Action playback requires recorded action values")
            _, reward, _, _ = self.env.step(action.values)
            success = bool(self.env._check_success())
        if self.viewer:
            self.env.render()
        if self._web_viewer is not None:
            self._web_viewer.update(self.env.sim)

        image = None
        if self.emit_images and action.episode_step % self._frame_stride == 0:
            image = self.env.sim.render(
                height=self.height,
                width=self.width,
                camera_name=self.camera,
            )[::-1]
        self._last_episode_step = action.episode_step
        speed = self.controls.snapshot().speed if self.controls is not None else 1.0
        self._next_step_at = max(
            self._next_step_at + (1.0 / (self.hz * speed)),
            time.monotonic(),
        )
        observation = RoboCasaObservation(
            image=image,
            image_updated=image is not None,
            source="robocasa",
            task=self.task,
            episode_step=action.episode_step,
            cycle=action.cycle,
            progress=action.episode_step / max(1, self._action_count - 1),
            reward=float(reward),
            success=success,
            action_norm=(
                float(np.linalg.norm(action.values))
                if action.values is not None
                else 0.0
            ),
        )
        return observation

    def finalize(self) -> None:
        cleanup_error: Exception | None = None
        if self._web_console is not None:
            try:
                self._web_console.close()
            except Exception as exc:  # noqa: BLE001 - preserve remaining cleanup
                cleanup_error = exc
        if self._web_viewer is not None:
            try:
                self._web_viewer.close()
            except Exception as exc:  # noqa: BLE001 - preserve remaining cleanup
                cleanup_error = cleanup_error or exc
        if self.env is not None:
            try:
                self.env.close()
            except Exception as exc:  # noqa: BLE001 - simulator cleanup is external
                cleanup_error = cleanup_error or exc
            finally:
                self.env = None
                print("Closed the Retriever-connected RoboCasa simulator.")
        if cleanup_error is not None:
            raise cleanup_error


class TaskVerifier(Flow[RoboCasaObservation, RoboCasaObservation]):
    """Publish explicit RoboCasa reward and success verification."""

    def __init__(self, *, controls: ReplayControls | None = None) -> None:
        self.controls = controls
        self._cycle: int | None = None
        self._latest: RoboCasaObservation | None = None

    def reset(self) -> None:
        self._cycle = None
        self._latest = None

    def _publish(self, observation: RoboCasaObservation) -> RoboCasaObservation:
        self._latest = observation
        if self.controls is not None:
            self.controls.update_observation(
                episode_step=observation.episode_step,
                cycle=observation.cycle,
                progress=observation.progress,
                reward=observation.reward,
                success=observation.success,
                action_norm=observation.action_norm,
            )
        return observation

    def step(self, observation: RoboCasaObservation) -> RoboCasaObservation:
        concrete = RoboCasaObservation(
            image=observation.image,
            image_updated=observation.image_updated,
            source=observation.source,
            task=observation.task,
            episode_step=observation.episode_step,
            cycle=observation.cycle,
            progress=observation.progress,
            reward=observation.reward,
            success=observation.success,
            action_norm=observation.action_norm,
        )
        if self._cycle is not None and concrete.cycle < self._cycle:
            return self._latest or concrete
        if self._cycle != concrete.cycle:
            self._cycle = concrete.cycle
            self._latest = None
        if self._latest is not None and concrete.episode_step < self._latest.episode_step:
            return self._latest
        concrete.progress = max(
            concrete.progress,
            self._latest.progress if self._latest is not None else 0.0,
        )
        return self._publish(concrete)


class EventSink(Flow[RoboCasaObservation, None]):
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
        should_print = (
            observation.episode_step % self.print_every == 0 or success_transition
        )
        if should_print and key != self._last_printed:
            print(
                f"[{observation.source} step={observation.episode_step:04d}] "
                f"progress={observation.progress:.1%} "
                f"reward={observation.reward:.3f} success={observation.success}"
            )
            self._last_printed = key
        self._last_success = observation.success


# Backward-compatible public name used by the original replay example.
ObservationPrinter = EventSink


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
    emit_images = (
        args.visualize == "rerun" or video_path is not None
    ) and not args.viewer
    controls = (
        ReplayControls(task=args.task, episode=args.episode)
        if args.visualize == "mjviser"
        else None
    )
    goal = EmbodiedGoal(
        text=getattr(args, "goal", "") or "",
        task=args.task,
        episode=args.episode,
        planner=getattr(args, "planner", "offline"),
        execution_mode=getattr(args, "execution_mode", "demonstration"),
    )
    planner = create_planner(goal.planner)
    initial_plan = planner.plan(goal)
    goal_source = GoalSource(initial_plan.goal)
    planning_flow = EmbodiedPlannerFlow(
        planner,
        initial_plan=initial_plan,
        planner_factory=create_planner,
    )
    if controls is not None:
        controls.configure_execution(initial_plan.goal, initial_plan)

        def submit_goal(submitted: EmbodiedGoal) -> SkillPlan:
            return planning_flow.step(submitted)

        controls.set_goal_handler(
            submit_goal,
            on_accept=lambda accepted_goal, _plan: goal_source.set_goal(
                accepted_goal
            ),
        )
    source = DemoActionSource(
        mode=args.mode,
        task=args.task,
        split=args.split,
        episode=args.episode,
        repeat=args.repeat,
        mock_steps=args.mock_steps,
        controls=controls,
    )
    simulator = RoboCasaSimulator(
        mode=args.mode,
        task=args.task,
        split=args.split,
        episode=args.episode,
        hz=args.hz,
        viewer=args.viewer,
        visualize=args.visualize,
        viser_host=getattr(args, "viser_host", "127.0.0.1"),
        viser_port=getattr(args, "viser_port", 8085),
        console_host=getattr(args, "console_host", "127.0.0.1"),
        console_port=getattr(args, "console_port", 8086),
        native_viser_controls=getattr(args, "native_viser_controls", False),
        emit_images=emit_images,
        camera=args.camera,
        width=args.width,
        height=args.height,
        image_hz=args.image_hz,
        mock_steps=args.mock_steps,
        controls=controls,
        open_browser=getattr(args, "open_browser", False),
        launch_id=getattr(args, "launch_id", ""),
    )
    verifier = TaskVerifier(controls=controls)
    event_sink = EventSink(print_every=args.print_every)

    pipeline = Pipeline("robocasa_demo_replay", on_lag="catch_up")
    with pipeline:
        goals = (goal_source @ Rate(hz=args.hz)).named("goal_source")
        planning = (
            planning_flow
            @ Trigger("text", "task", "episode", "planner", "execution_mode")
        ).named(
            "embodied_planner"
        )
        dispatch = (SkillDispatcher() @ Trigger("goal")).named("skill_dispatcher")
        actions = (source @ Rate(hz=args.hz, on_lag="catch_up")).named(
            "demo_actions"
        )
        simulation = (simulator @ Trigger("cycle", "episode_step")).named(
            "robocasa_simulator"
        )
        verification = (verifier @ Trigger("episode_step")).named("task_verifier")
        output = (event_sink @ Trigger("episode_step")).named("event_sink")
        goals.then(planning, sync=Latest())
        planning.then(dispatch, sync=Latest())
        dispatch.then(actions, sync=Latest())
        actions.then(simulation, sync=Latest())
        simulation.then(verification, sync=Latest())
        verification.then(output, sync=Latest())
        if video_path is not None:
            video = (
                VideoRecorder(
                    path=video_path,
                    fps=getattr(args, "video_fps", args.image_hz),
                )
                @ Trigger("episode_step")
            ).named("video_recorder")
            verification.then(video, sync=Latest())
    return pipeline, simulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Retriever-connected RoboCasa demonstration replay."
    )
    parser.add_argument("--mode", choices=["mock", "robocasa"], default="mock")
    parser.add_argument("--task", default="TurnOnMicrowave")
    parser.add_argument("--split", choices=["pretrain", "target"], default="pretrain")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--goal", default="")
    parser.add_argument("--planner", choices=["offline", "gemini"], default="offline")
    parser.add_argument(
        "--execution-mode",
        choices=["demonstration", "live_planning"],
        default="demonstration",
        help=(
            "Replay a pinned demonstrated plan or plan allow-listed skills at "
            "goal submission time; both modes execute recorded replay data and "
            "report RoboCasa task verification."
        ),
    )
    parser.add_argument("--seconds", type=float, default=45.0)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--mock-steps", type=int, default=12)
    parser.add_argument("--hz", type=float, default=20.0)
    parser.add_argument("--image-hz", type=float, default=5.0)
    parser.add_argument("--camera", default="robot0_agentview_center")
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--visualize",
        choices=["none", "rerun", "mjviser"],
        default="none",
    )
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8085)
    parser.add_argument("--console-host", default="127.0.0.1")
    parser.add_argument("--console-port", type=int, default=8086)
    parser.add_argument("--launch-id", default="")
    parser.add_argument(
        "--native-viser-controls",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also expose Retriever controls inside mjviser for debugging.",
    )
    parser.add_argument(
        "--open-browser",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Open the Retriever console after the simulator scene is ready.",
    )
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


def _execute(args: argparse.Namespace) -> None:
    if args.viewer and args.video:
        raise SystemExit(
            "--video uses the offscreen MuJoCo renderer and cannot be combined "
            "with --viewer on macOS. Run them as separate commands."
        )
    if args.viewer and args.visualize == "mjviser":
        raise SystemExit("Use either --viewer or --visualize mjviser, not both.")
    if args.mode == "mock" and args.visualize == "mjviser":
        raise SystemExit("mjviser requires --mode robocasa with a real MuJoCo scene.")
    retriever.init(default_viz=VizConfig(hz=args.image_hz, fields=None))
    pipeline, simulator = build_pipeline(args)

    if args.visualize == "rerun" and args.rerun_mode == "record":
        from retriever.lib.rerun import record_session

        Path(args.recording).parent.mkdir(parents=True, exist_ok=True)
        max_steps = (
            args.steps if args.mode == "mock" else max(1, round(args.seconds * args.hz))
        )
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
    if args.visualize == "rerun":
        pipeline.run(
            backend="in-process",
            duration=args.seconds,
            visualize="rerun",
            blocking=True,
            backend_config={
                "rerun_config": {
                    "spawn": args.rerun_mode == "spawn",
                    "connect_addr": args.rerun_address,
                }
            },
        )
        return

    # Step-counted execution keeps expensive RoboCasa initialization outside the
    # requested replay duration and leaves the console alive for restart controls.
    try:
        for _ in range(max(1, round(args.seconds * args.hz))):
            pipeline.step(dt=1.0 / args.hz)
            simulator.refresh_controls()
    finally:
        pipeline.close_stepper()


def run(
    *,
    task: str = "PrepareCoffee",
    episode: int = 0,
    planner: str = "offline",
    execution_mode: str = "demonstration",
    visualize: str = "mjviser",
    open_browser: bool = True,
    goal: str = "",
    split: str = "pretrain",
    seconds: float = 120.0,
    hz: float = 20.0,
) -> None:
    """Run one RoboCasa demonstration with task verification in the console."""

    args = argparse.Namespace(
        mode="robocasa",
        task=task,
        split=split,
        episode=episode,
        goal=goal,
        planner=planner,
        execution_mode=execution_mode,
        seconds=seconds,
        steps=14,
        mock_steps=12,
        hz=hz,
        image_hz=5.0,
        camera="robot0_agentview_center",
        width=768,
        height=512,
        viewer=False,
        visualize=visualize,
        viser_host="127.0.0.1",
        viser_port=8085,
        console_host="127.0.0.1",
        console_port=8086,
        launch_id="",
        native_viser_controls=False,
        open_browser=open_browser,
        rerun_mode="spawn",
        rerun_address="127.0.0.1:9876",
        recording="logs/robocasa-replay.rrd",
        video=None,
        video_fps=5.0,
        repeat=False,
        print_every=100,
    )
    _execute(args)


def main() -> None:
    _execute(parse_args())


if __name__ == "__main__":
    main()
