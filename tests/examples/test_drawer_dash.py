"""Mock-lane tests for the RoboCasa drawer-dash example.

These run in the default environment, which has neither MuJoCo nor RoboCasa
installed, so they double as the guard that the example stays import-safe.
"""

import sys
from argparse import Namespace

import pytest
from examples.advanced.robocasa_drawer_dash import plan
from examples.advanced.robocasa_drawer_dash.app import (
    ChoreographyPolicy,
    DrawerDashAction,
    DrawerDashSimulator,
    build_pipeline,
)


def _mock_args(**overrides: object) -> Namespace:
    values = {
        "mode": "mock",
        "scene": None,
        "camera": "threequarter",
        "steps": 38,
        "dt": 0.5,
        "hz": 2.0,
        "print_every": 100,
    }
    values.update(overrides)
    return Namespace(**values)


def _run(args: Namespace) -> DrawerDashSimulator:
    pipeline, simulator = build_pipeline(args)
    try:
        for _ in range(args.steps):
            pipeline.step(dt=args.dt)
    finally:
        pipeline.close_stepper()
    return simulator


def test_importing_the_example_does_not_pull_in_a_simulator() -> None:
    # AGENTS.md: every example lane must run without hardware, models or GPUs.
    assert "mujoco" not in sys.modules
    assert "robocasa" not in sys.modules
    assert "robosuite" not in sys.modules


def test_mock_routine_opens_and_shuts_the_drawer_without_a_simulator() -> None:
    simulator = _run(_mock_args())
    latest = simulator.latest

    assert latest is not None
    assert latest.source == "mock"
    assert latest.done is True
    assert latest.success is True
    assert latest.peak_open >= plan.OPENED_MIN
    assert latest.drawer_open <= plan.SHUT_MAX


def test_mock_lane_is_deterministic() -> None:
    first = _run(_mock_args()).latest
    second = _run(_mock_args()).latest

    assert first is not None and second is not None
    assert first.peak_open == pytest.approx(second.peak_open)
    assert first.drawer_open == pytest.approx(second.drawer_open)
    assert first.phase == second.phase


def test_drawer_only_moves_while_the_handle_is_grasped() -> None:
    # The whole point of the scene: the slide joint carries no actuator, so a
    # tick that is not holding the handle must not move the drawer.
    simulator = DrawerDashSimulator(mode="mock", scene=None, camera="threequarter", hz=2.0)
    simulator.reset()

    opened = simulator.step(
        DrawerDashAction(phase="pull drawer open", phase_index=4, blend=1.0,
                         grip=plan.SQUEEZE, stroke=plan.STROKE, grasping=True, elapsed=8.0)
    )
    assert opened.drawer_open == pytest.approx(plan.STROKE)

    # Same commanded stroke, but nothing is holding the bar.
    released = simulator.step(
        DrawerDashAction(phase="line up", phase_index=1, blend=1.0,
                         grip=plan.OPEN, stroke=plan.STROKE, grasping=False, elapsed=9.0)
    )
    assert released.grasped is False
    assert released.drawer_open == pytest.approx(plan.STROKE)  # holds, does not advance


def test_policy_walks_every_phase_in_order() -> None:
    policy = ChoreographyPolicy(hz=20.0)
    policy.reset()

    seen: list[str] = []
    for _ in range(int(plan.TOTAL_SECONDS * 20.0)):
        action = policy.step(None)
        if not seen or seen[-1] != action.phase:
            seen.append(action.phase)

    assert seen == [phase.label for phase in plan.PHASES]


def test_phase_schedule_is_contiguous_and_clamps_past_the_end() -> None:
    assert plan.TOTAL_SECONDS == pytest.approx(sum(p.seconds for p in plan.PHASES))

    index, phase, blend = plan.phase_at(0.0)
    assert (index, phase.label, blend) == (0, "settle", 0.0)

    # Overrunning the routine holds the final phase rather than wrapping round.
    index, phase, blend = plan.phase_at(plan.TOTAL_SECONDS + 5.0)
    assert index == len(plan.PHASES) - 1
    assert phase.label == "withdraw"
    assert blend == pytest.approx(1.0)


def test_grasping_phases_are_the_ones_that_command_a_squeeze() -> None:
    for phase in plan.PHASES:
        if phase.label in plan.GRASPING:
            assert phase.grip == plan.SQUEEZE
        else:
            assert phase.grip == plan.OPEN


def test_mujoco_lane_explains_itself_when_the_scene_is_missing() -> None:
    simulator = DrawerDashSimulator(
        mode="mujoco", scene="/nonexistent/scene.xml", camera="threequarter", hz=20.0
    )
    with pytest.raises(RuntimeError) as excinfo:
        simulator.reset()

    message = str(excinfo.value)
    assert "demo-drawer-dash-mock" in message or "scene" in message
