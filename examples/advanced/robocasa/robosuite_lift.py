"""Basic robosuite Lift loop expressed as Retriever Flows.

Run the mock-safe smoke test:
  pixi run demo-robosuite-mock

Run against robosuite through the locked simulator environment:
  pixi run --locked -e robocasa demo-robosuite-lift
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io

from examples.advanced.openpi_policy.common import ActionChunk, PolicyObservation

from .method_harness import (
    HarnessEvent,
    MethodHarness,
    SafetyEnvelope,
    Transition,
    TrialRequest,
    Verification,
)
from .mjviser_bridge import MjviserBridge


@io
class LiftAction:
    dx: float | None = None
    dy: float | None = None
    dz: float | None = None
    grip: float | None = None


@io
class LiftState:
    step: int | None = None
    source: str | None = None
    object_x: float | None = None
    object_y: float | None = None
    object_height: float | None = None
    gripper_x: float | None = None
    gripper_y: float | None = None
    gripper_z: float | None = None
    grasped: bool | None = None
    reward: float | None = None
    grasped: bool | None = None
    done: bool | None = None


class LiftEnvFlow(Flow[LiftAction, LiftState]):
    """Small env wrapper that can use robosuite or a deterministic mock."""

    def __init__(
        self,
        *,
        mode: str,
        env_name: str,
        robot: str,
        has_renderer: bool,
        visualize: str = "none",
        viser_host: str = "127.0.0.1",
        viser_port: int = 8085,
        seed: int | None = 0,
    ) -> None:
        super().__init__()
        self.mode = mode
        self.env_name = env_name
        self.robot = robot
        self.has_renderer = has_renderer
        self.visualize = visualize
        self.seed = seed
        self._web_viewer = (
            MjviserBridge(
                host=viser_host,
                port=viser_port,
                label=f"Retriever robosuite {env_name}",
                camera_preset="Robot",
                robot_oriented_camera=False,
            )
            if visualize == "mjviser"
            else None
        )

    def init(self) -> None:
        self.step_idx = 0
        self._env: Any | None = None
        self._obs: dict[str, Any] = {}
        self._mock_object_height = 0.82
        self._mock_object_x = 0.0
        self._mock_object_y = 0.0
        self._mock_gripper_z = 0.96
        self._mock_gripper_x = 0.10
        self._mock_gripper_y = 0.10
        self._mock_lift_started = False

        if self.mode == "robosuite":
            try:
                import robosuite as suite
            except ImportError as exc:
                raise RuntimeError(
                    "robosuite is not installed. Run "
                    "`pixi install --locked -e robocasa` "
                    "for the real simulator or `pixi run demo-robosuite-mock` "
                    "for the mock-safe smoke test."
                ) from exc

            self._env = suite.make(
                env_name=self.env_name,
                robots=self.robot,
                has_renderer=self.has_renderer,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                control_freq=20,
                ignore_done=self.visualize == "mjviser",
                seed=self.seed,
            )
            self._obs = self._env.reset()
            if self._web_viewer is not None:
                self._web_viewer.update(self._env.sim)

    def step(self, action: LiftAction | None) -> LiftState:
        self.step_idx += 1
        dx = 0.0 if action is None or action.dx is None else float(action.dx)
        dy = 0.0 if action is None or action.dy is None else float(action.dy)
        dz = 0.0 if action is None or action.dz is None else float(action.dz)
        grip = -1.0 if action is None or action.grip is None else float(action.grip)

        if self.mode == "robosuite":
            return self._step_robosuite(dx=dx, dy=dy, dz=dz, grip=grip)
        return self._step_mock(dz=dz, grip=grip)

    def _step_mock(self, *, dz: float, grip: float) -> LiftState:
        near_object = self._mock_gripper_z <= self._mock_object_height + 0.10
        if grip > 0.2 and near_object:
            self._mock_lift_started = True

        self._mock_gripper_x += dx * 0.04
        self._mock_gripper_y += dy * 0.04
        self._mock_gripper_z = max(0.75, min(1.25, self._mock_gripper_z + dz * 0.04))
        if self._mock_lift_started:
            self._mock_object_height = min(
                1.18, self._mock_object_height + max(dz, 0.0) * 0.035
            )
        reward = max(0.0, self._mock_object_height - 0.82)
        done = self._mock_object_height >= 1.05
        return LiftState(
            step=self.step_idx,
            source="mock",
            object_x=self._mock_object_x,
            object_y=self._mock_object_y,
            object_height=self._mock_object_height,
            gripper_x=self._mock_gripper_x,
            gripper_y=self._mock_gripper_y,
            gripper_z=self._mock_gripper_z,
            grasped=self._mock_lift_started,
            reward=reward,
            grasped=self._mock_lift_started,
            done=done,
        )

    def _step_robosuite(
        self, *, dx: float, dy: float, dz: float, grip: float
    ) -> LiftState:
        if self._env is None:
            raise RuntimeError("robosuite environment was not initialized")

        low, high = self._env.action_spec
        control = np.zeros_like(low, dtype=float)
        if control.size >= 3:
            control[:3] = (dx, dy, dz)
        if control.size >= 1:
            control[-1] = grip
        control = np.clip(control, low, high)

        result = self._env.step(control)
        if len(result) == 5:
            obs, reward, terminated, truncated, _info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _info = result
        done = bool(done or float(reward) > 0.0)
        self._obs = obs
        check_success = getattr(self._env, "_check_success", None)
        task_success = bool(check_success()) if callable(check_success) else False
        check_grasp = getattr(self._env, "_check_grasp", None)
        grasped = False
        if callable(check_grasp):
            grasped = bool(
                check_grasp(
                    gripper=self._env.robots[0].gripper,
                    object_geoms=self._env.cube,
                )
            )
        done = bool(done or task_success)
        if self._web_viewer is not None:
            self._web_viewer.update(self._env.sim)

        object_pos = _safe_xyz(obs, "cube_pos")
        gripper_pos = _safe_xyz(obs, "robot0_eef_pos")
        grasped = self._env._check_grasp(self._env.robots[0].gripper, self._env.cube)
        return LiftState(
            step=self.step_idx,
            source="robosuite",
            object_x=_axis(object_pos, 0),
            object_y=_axis(object_pos, 1),
            object_height=_axis(object_pos, 2),
            gripper_x=_axis(gripper_pos, 0),
            gripper_y=_axis(gripper_pos, 1),
            gripper_z=_axis(gripper_pos, 2),
            grasped=bool(grasped),
            reward=float(reward),
            done=done,
        )

    def finalize(self) -> None:
        if self._web_viewer is not None:
            self._web_viewer.close()
        if self._env is not None:
            self._env.close()


def _safe_xyz(obs: dict[str, Any], key: str) -> tuple[float, float, float] | None:
    value = obs.get(key)
    if value is None or len(value) < 3:
        return None
    return float(value[0]), float(value[1]), float(value[2])


def _axis(value: tuple[float, float, float] | None, index: int) -> float | None:
    return None if value is None else value[index]


def _clamp(value: float) -> float:
    return max(-1.0, min(1.0, value))


class HeuristicLiftPolicy(Flow[LiftState, LiftAction]):
    """Bounded scripted policy: align, descend, grasp, then lift."""

    def __init__(self, *, target_height: float) -> None:
        super().__init__()
        self.target_height = target_height

    def init(self) -> None:
        self.phase = "approach"
        self.grasp_ticks = 0
        self.grasp_offset = 0.015

    def step(self, state: LiftState | None) -> LiftAction:
        if state is None or state.object_height is None or state.gripper_z is None:
            return LiftAction(dz=-0.4, grip=-1.0)
        if state.object_height >= self.target_height:
            return LiftAction(dz=0.0, grip=1.0)
        if None not in (
            state.object_x,
            state.object_y,
            state.gripper_x,
            state.gripper_y,
        ):
            dx = float(state.object_x) - float(state.gripper_x)
            dy = float(state.object_y) - float(state.gripper_y)
            if abs(dx) > 0.004 or abs(dy) > 0.004:
                return LiftAction(
                    dx=_clamp(dx * 10.0),
                    dy=_clamp(dy * 10.0),
                    dz=-0.2,
                    grip=-1.0,
                )
        if state.grasped:
            return LiftAction(dz=1.0, grip=1.0)
        if state.gripper_z > state.object_height + 0.06:
            return LiftAction(dz=-0.5, grip=-1.0)
        if state.gripper_z > state.object_height + 0.005:
            return LiftAction(dz=-0.2, grip=-1.0)
        return LiftAction(dz=0.0, grip=1.0)


class LiftPrinter(Flow[LiftState, None]):
    def __init__(self, *, print_every: int) -> None:
        super().__init__()
        self.print_every = max(1, int(print_every))

    def step(self, state: LiftState) -> None:
        if state.step is None or state.step % self.print_every != 0:
            return
        print(
            f"[{state.source} step={state.step:03d}] "
            f"object_z={_fmt(state.object_height)} "
            f"gripper_z={_fmt(state.gripper_z)} "
            f"reward={_fmt(state.reward)} done={bool(state.done)}"
        )


def _policy_observation(state: LiftState) -> PolicyObservation:
    values = (
        state.object_x,
        state.object_y,
        state.object_height,
        state.gripper_x,
        state.gripper_y,
        state.gripper_z,
        state.reward,
    )
    if any(value is None for value in values):
        raise RuntimeError("Lift environment returned an incomplete privileged state")
    return PolicyObservation(
        image=np.zeros((1, 1, 3), dtype=np.uint8),
        state=np.asarray(
            [*values, float(bool(state.grasped))],
            dtype=np.float32,
        ),
        prompt="Lift the cube above the table",
    )


class LiftHarnessEnvironment:
    """Adapt the Retriever Lift Flow to the generic methods harness."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.flow = LiftEnvFlow(
            mode=args.mode,
            env_name=args.env,
            robot=args.robot,
            has_renderer=args.viewer,
            visualize=args.visualize,
            viser_host=args.viser_host,
            viser_port=args.viser_port,
        )
        self.last_state: LiftState | None = None

    def reset(self, request: TrialRequest) -> PolicyObservation:
        np.random.seed(request.seed)
        self.flow.seed = request.seed
        self.flow.init()
        self.last_state = self.flow.step(LiftAction(grip=-1.0))
        return _policy_observation(self.last_state)

    def step(self, action: np.ndarray) -> Transition:
        values = np.asarray(action, dtype=np.float32)
        if values.shape != (4,):
            raise ValueError("Lift actions must have shape (4,)")
        self.last_state = self.flow.step(
            LiftAction(
                dx=float(values[0]),
                dy=float(values[1]),
                dz=float(values[2]),
                grip=float(values[3]),
            )
        )
        if self.args.visualize == "mjviser":
            time.sleep(self.args.dt)
        return Transition(
            observation=_policy_observation(self.last_state),
            reward=float(self.last_state.reward or 0.0),
            terminated=bool(self.last_state.done),
        )

    def verify(self) -> Verification:
        state = self.last_state or LiftState()
        success = bool(state.done)
        return Verification(
            success=success,
            reward=float(state.reward or 0.0),
            message=(
                "robosuite native task success"
                if success
                else "robosuite native task success was not reached"
            ),
        )

    def close(self) -> None:
        self.flow.finalize()


class ScriptedLiftMethod:
    """Produce bounded four-step future-action chunks from privileged state."""

    def __init__(self, *, target_height: float) -> None:
        self.policy = HeuristicLiftPolicy(target_height=target_height)
        self.policy.init()

    def predict(self, observation: PolicyObservation) -> ActionChunk:
        state = np.asarray(observation.state, dtype=np.float32)
        if state.shape != (8,):
            raise ValueError("Lift policy state must have shape (8,)")
        action = self.policy.step(
            LiftState(
                object_x=float(state[0]),
                object_y=float(state[1]),
                object_height=float(state[2]),
                gripper_x=float(state[3]),
                gripper_y=float(state[4]),
                gripper_z=float(state[5]),
                reward=float(state[6]),
                grasped=bool(state[7]),
            )
        )
        row = np.asarray(
            [
                action.dx or 0.0,
                action.dy or 0.0,
                action.dz or 0.0,
                action.grip or 0.0,
            ],
            dtype=np.float32,
        )
        return ActionChunk(
            actions=np.repeat(row[None, :], 4, axis=0),
            horizon=4,
            dof=4,
            source="scripted",
        )


def run_harness(args: argparse.Namespace) -> None:
    environment = LiftHarnessEnvironment(args)

    def print_event(event: HarnessEvent) -> None:
        if event.kind in {"chunk_dispatched", "verification", "trial_error"}:
            print(
                f"[harness {event.kind} step={event.step:03d}] "
                f"{event.status}: {event.message}"
            )

    started_at = time.monotonic()
    try:
        report = MethodHarness(
            safety=SafetyEnvelope(
                max_horizon=4,
                max_dof=4,
                max_abs_action=1.0,
                allowed_sources=frozenset({"scripted"}),
            ),
            event_sink=print_event,
        ).run(
            TrialRequest(
                method_id="scripted-privileged-lift",
                task=args.env,
                max_steps=args.steps,
            ),
            environment=environment,
            method=ScriptedLiftMethod(target_height=args.target_height),
            close_environment=False,
        )
        print(
            f"[harness report] status={report.status} success={report.success} "
            f"steps={report.steps} reward={report.verification.reward:.3f}"
        )
        if args.visualize == "mjviser":
            remaining = args.steps * args.dt - (time.monotonic() - started_at)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        environment.close()


def _fmt(value: float | None) -> str:
    return "None" if value is None else f"{value:.3f}"


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    pipe = Pipeline("robosuite_lift_demo")
    with pipe:
        env = LiftEnvFlow(
            mode=args.mode,
            env_name=args.env,
            robot=args.robot,
            has_renderer=args.viewer,
            visualize=args.visualize,
            viser_host=args.viser_host,
            viser_port=args.viser_port,
            seed=args.seed,
        ) @ Rate(hz=args.env_hz)
        policy = HeuristicLiftPolicy(target_height=args.target_height) @ Rate(
            hz=args.policy_hz
        )
        printer = LiftPrinter(print_every=args.print_every) @ Trigger("step")
        pipe.connect(env, policy, sync=Latest())
        pipe.connect(policy, env, sync=Latest())
        pipe.connect(env, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Basic Retriever + robosuite Lift demo."
    )
    parser.add_argument("--mode", choices=["mock", "robosuite"], default="mock")
    parser.add_argument("--env", default="Lift")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument(
        "--viewer", action="store_true", help="Enable robosuite's native viewer."
    )
    parser.add_argument(
        "--visualize",
        choices=["none", "mjviser"],
        default="none",
        help="Stream the live MuJoCo scene to a browser with mjviser.",
    )
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8085)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--env-hz", type=float, default=20.0)
    parser.add_argument("--policy-hz", type=float, default=5.0)
    parser.add_argument("--target-height", type=float, default=1.05)
    parser.add_argument("--print-every", type=int, default=2)
    parser.add_argument(
        "--harness",
        action="store_true",
        help="Execute through the typed methods harness and ActionChunk boundary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.viewer and args.visualize == "mjviser":
        raise SystemExit("Use either --viewer or --visualize mjviser, not both.")
    if args.mode == "mock" and args.visualize == "mjviser":
        raise SystemExit("mjviser requires --mode robosuite with a real MuJoCo scene.")
    if args.harness:
        run_harness(args)
        return
    pipe = build_pipeline(args)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
            if args.visualize == "mjviser":
                time.sleep(args.dt)
    finally:
        pipe.close_stepper()


if __name__ == "__main__":
    main()
