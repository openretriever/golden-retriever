"""Basic robosuite Lift loop expressed as Retriever Flows.

Run the mock-safe smoke test:
  pixi run demo-robosuite-mock

Run against robosuite after installing the optional dependency:
  pixi run python -m pip install -e ".[robosuite]"
  pixi run demo-robosuite-lift
"""

from __future__ import annotations

import argparse
from typing import Any

from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, io


@io
class LiftAction:
    dz: float | None = None
    grip: float | None = None


@io
class LiftState:
    step: int | None = None
    source: str | None = None
    object_height: float | None = None
    gripper_z: float | None = None
    reward: float | None = None
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
    ) -> None:
        super().__init__()
        self.mode = mode
        self.env_name = env_name
        self.robot = robot
        self.has_renderer = has_renderer

    def init(self) -> None:
        self.step_idx = 0
        self._env: Any | None = None
        self._obs: dict[str, Any] = {}
        self._mock_object_height = 0.82
        self._mock_gripper_z = 0.96
        self._mock_lift_started = False

        if self.mode == "robosuite":
            try:
                import robosuite as suite
            except ImportError as exc:
                raise RuntimeError(
                    "robosuite is not installed. Install the optional dependency with "
                    '`pixi run python -m pip install -e ".[robosuite]"` or run '
                    "`pixi run demo-robosuite-mock` for the mock-safe smoke test."
                ) from exc

            self._env = suite.make(
                env_name=self.env_name,
                robots=self.robot,
                has_renderer=self.has_renderer,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                control_freq=20,
            )
            self._obs = self._env.reset()

    def step(self, action: LiftAction | None) -> LiftState:
        self.step_idx += 1
        dz = 0.0 if action is None or action.dz is None else float(action.dz)
        grip = 1.0 if action is None or action.grip is None else float(action.grip)

        if self.mode == "robosuite":
            return self._step_robosuite(dz=dz, grip=grip)
        return self._step_mock(dz=dz, grip=grip)

    def _step_mock(self, *, dz: float, grip: float) -> LiftState:
        near_object = self._mock_gripper_z <= self._mock_object_height + 0.10
        if grip < -0.2 and near_object:
            self._mock_lift_started = True

        self._mock_gripper_z = max(0.75, min(1.25, self._mock_gripper_z + dz * 0.04))
        if self._mock_lift_started:
            self._mock_object_height = min(1.18, self._mock_object_height + max(dz, 0.0) * 0.035)
        reward = max(0.0, self._mock_object_height - 0.82)
        done = self._mock_object_height >= 1.05
        return LiftState(
            step=self.step_idx,
            source="mock",
            object_height=self._mock_object_height,
            gripper_z=self._mock_gripper_z,
            reward=reward,
            done=done,
        )

    def _step_robosuite(self, *, dz: float, grip: float) -> LiftState:
        if self._env is None:
            raise RuntimeError("robosuite environment was not initialized")

        import numpy as np

        low, high = self._env.action_spec
        control = np.zeros_like(low, dtype=float)
        if control.size >= 3:
            control[2] = dz
        if control.size >= 1:
            control[-1] = grip
        control = np.clip(control, low, high)

        result = self._env.step(control)
        if len(result) == 5:
            obs, reward, terminated, truncated, _info = result
            done = bool(terminated or truncated)
        else:
            obs, reward, done, _info = result
        self._obs = obs

        object_height = _safe_z(obs, "cube_pos")
        gripper_z = _safe_z(obs, "robot0_eef_pos")
        return LiftState(
            step=self.step_idx,
            source="robosuite",
            object_height=object_height,
            gripper_z=gripper_z,
            reward=float(reward),
            done=bool(done),
        )

    def finalize(self) -> None:
        if self._env is not None:
            self._env.close()


def _safe_z(obs: dict[str, Any], key: str) -> float | None:
    value = obs.get(key)
    if value is None or len(value) < 3:
        return None
    return float(value[2])


class HeuristicLiftPolicy(Flow[LiftState, LiftAction]):
    """Tiny scripted policy: approach, close gripper, then lift."""

    def __init__(self, *, target_height: float) -> None:
        super().__init__()
        self.target_height = target_height

    def step(self, state: LiftState | None) -> LiftAction:
        if state is None or state.object_height is None or state.gripper_z is None:
            return LiftAction(dz=-0.4, grip=1.0)
        if state.done or state.object_height >= self.target_height:
            return LiftAction(dz=0.0, grip=-1.0)
        if state.gripper_z > state.object_height + 0.08:
            return LiftAction(dz=-0.5, grip=1.0)
        return LiftAction(dz=0.6, grip=-1.0)


class LiftPrinter(Flow[LiftState, None]):
    def __init__(self, *, print_every: int) -> None:
        super().__init__()
        self.print_every = max(1, int(print_every))

    def step(self, state: LiftState) -> None:
        if state.step is None or state.step % self.print_every != 0:
            return None
        print(
            f"[{state.source} step={state.step:03d}] "
            f"object_z={_fmt(state.object_height)} "
            f"gripper_z={_fmt(state.gripper_z)} "
            f"reward={_fmt(state.reward)} done={bool(state.done)}"
        )
        return None


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
        ) @ Rate(hz=args.env_hz)
        policy = HeuristicLiftPolicy(target_height=args.target_height) @ Rate(hz=args.policy_hz)
        printer = LiftPrinter(print_every=args.print_every) @ Trigger("step")
        pipe.connect(env, policy, sync=Latest())
        pipe.connect(policy, env, sync=Latest())
        pipe.connect(env, printer, sync=Latest())
    return pipe


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Basic Retriever + robosuite Lift demo.")
    parser.add_argument("--mode", choices=["mock", "robosuite"], default="mock")
    parser.add_argument("--env", default="Lift")
    parser.add_argument("--robot", default="Panda")
    parser.add_argument("--viewer", action="store_true", help="Enable robosuite's native viewer.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--env-hz", type=float, default=20.0)
    parser.add_argument("--policy-hz", type=float, default=5.0)
    parser.add_argument("--target-height", type=float, default=1.05)
    parser.add_argument("--print-every", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pipe = build_pipeline(args)
    try:
        for _ in range(args.steps):
            pipe.step(dt=args.dt)
    finally:
        pipe.close_stepper()


if __name__ == "__main__":
    main()
