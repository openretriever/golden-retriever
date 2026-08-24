from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.advanced.robosuite_lift.app import HeuristicLiftPolicy, LiftEnvFlow


class RobosuiteLiftMockTests(unittest.TestCase):
    """Deterministic mock-mode regression test for the robosuite_lift demo.

    Needs no simulator dependency (mode="mock" is a hand-rolled physics
    stand-in), so this always runs, unlike the real-robosuite path which
    requires the optional `robosuite` extra."""

    def test_heuristic_policy_lifts_the_mock_cube_to_target_height(self) -> None:
        env = LiftEnvFlow(mode="mock", env_name="Lift", robot="Panda", has_renderer=False)
        env.init()
        policy = HeuristicLiftPolicy(target_height=1.05)

        state = env.step(None)
        reached_target = False
        for _ in range(200):
            action = policy.step(state)
            state = env.step(action)
            if state.object_height is not None and state.object_height >= 1.05:
                reached_target = True
                break

        self.assertTrue(reached_target, "heuristic policy never lifted the mock cube to the target height")
        self.assertTrue(state.done)


if __name__ == "__main__":
    unittest.main()
