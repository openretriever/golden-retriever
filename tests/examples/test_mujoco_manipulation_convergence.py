from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MUJOCO_MANIPULATION_DIR = ROOT / "examples" / "advanced" / "mujoco_manipulation"
for _path in (ROOT, MUJOCO_MANIPULATION_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

pytest.importorskip("mujoco", reason="mujoco is an optional simulation dependency")

from flows import Control, ControllerFlow, MujocoEnvFlow  # noqa: E402


class MujocoManipulationConvergenceTests(unittest.TestCase):
    """Drives the real MuJoCo physics headlessly (no render, no Rerun/dora) to
    turn the mujoco_manipulation demo's "does the arm chase the target" claim
    into a pass/fail check instead of an eyeballed Rerun animation."""

    def _run(self, steps: int, *, controlled: bool) -> np.ndarray:
        env_flow = MujocoEnvFlow()
        env_flow.init()
        # Constructing a renderer needs a GL context; this test only checks
        # physics, so make sure render() is never triggered.
        env_flow.render_every = steps + 1

        controller = ControllerFlow() if controlled else None
        if controller is not None:
            controller.init()

        state = env_flow.run(Control(ctrl=np.zeros(2)))
        distances = []
        for _ in range(steps):
            ctrl = controller.run(state) if controller is not None else Control(ctrl=np.zeros(2))
            state = env_flow.run(ctrl)
            distances.append(float(np.linalg.norm(state.target_pos[:2] - state.tip_pos[:2])))
        return np.array(distances)

    def test_controller_tracks_moving_target_far_better_than_no_control(self) -> None:
        steps = 4000  # 20s of sim time at the model's 5ms physics step, ~3 orbits of the target
        skip = 1000  # ignore the first 5s transient from rest

        controlled = self._run(steps, controlled=True)
        uncontrolled = self._run(steps, controlled=False)

        self.assertTrue(np.all(np.isfinite(controlled)), "physics went unstable (NaN/Inf)")

        controlled_mean = float(controlled[skip:].mean())
        uncontrolled_mean = float(uncontrolled[skip:].mean())

        # Absolute bound: tracking error should stay within a modest fraction
        # of the arm's own 1.0m reach, not merely oscillate near its start pose.
        self.assertLess(controlled_mean, 0.15)
        self.assertLess(float(controlled[skip:].max()), 0.25)

        # Relative bound: the controller must do meaningfully better than zero
        # torque, to catch a silently disconnected/inverted feedback loop that
        # would still produce finite, bounded-looking output.
        self.assertLess(controlled_mean, 0.5 * uncontrolled_mean)


if __name__ == "__main__":
    unittest.main()
