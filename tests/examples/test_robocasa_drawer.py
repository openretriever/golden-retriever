"""Mock-lane tests for the RoboCasa drawer example.

These run in the default environment, which has neither MuJoCo nor RoboCasa
installed, so they double as the guard that the example stays import-safe.
"""

import sys
from argparse import Namespace

import pytest
from examples.advanced.robocasa_drawer import plan
from examples.advanced.robocasa_drawer.app import (
    MOCK_STEPS,
    ChoreographyPolicy,
    DrawerAction,
    DrawerSimulator,
    build_pipeline,
)


def _mock_args(**overrides: object) -> Namespace:
    values = {
        "mode": "mock",
        "scene": None,
        "camera": "action",
        "steps": MOCK_STEPS,
        "dt": 0.5,
        "hz": 2.0,
        "print_every": 1000,
    }
    values.update(overrides)
    return Namespace(**values)


def _run(args: Namespace) -> DrawerSimulator:
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


def test_mock_routine_opens_the_drawer_seasons_the_plate_and_shuts_it() -> None:
    simulator = _run(_mock_args())
    latest = simulator.latest

    assert latest is not None
    assert latest.source == "mock"
    assert latest.done is True
    assert latest.success is True
    assert latest.peak_open >= plan.OPENED_MIN
    assert latest.drawer_open <= plan.SHUT_MAX
    assert latest.seasoned is True
    assert latest.stowed is True


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
    simulator = DrawerSimulator(mode="mock", scene=None, camera="action", hz=2.0)
    simulator.reset()

    opened = simulator.step(
        DrawerAction(phase="pull the drawer open", phase_index=4, blend=1.0,
                     grip=plan.HANDLE_SQUEEZE, stroke=plan.STROKE, tip=0.0,
                     grasping=True, carrying=False, elapsed=8.0)
    )
    assert opened.drawer_open == pytest.approx(plan.STROKE)

    # Same commanded stroke, but nothing is holding the bar.
    released = simulator.step(
        DrawerAction(phase="back off the handle", phase_index=6, blend=1.0,
                     grip=plan.OPEN, stroke=0.0, tip=0.0,
                     grasping=False, carrying=False, elapsed=9.0)
    )
    assert released.grasped is False
    assert released.drawer_open == pytest.approx(plan.STROKE)  # holds, does not shut


def test_the_shaker_is_only_seasoning_while_it_is_held() -> None:
    # Tipping a shaker it is not holding proves nothing, so the mock only
    # counts the roll while the grasp is on.
    simulator = DrawerSimulator(mode="mock", scene=None, camera="action", hz=2.0)
    simulator.reset()

    loose = simulator.step(
        DrawerAction(phase="over the shaker", phase_index=8, blend=1.0,
                     grip=plan.OPEN, stroke=plan.STROKE, tip=plan.TIP_ANGLE,
                     grasping=False, carrying=False, elapsed=14.0)
    )
    assert loose.seasoned is False

    held = simulator.step(
        DrawerAction(phase="tip it over the food", phase_index=14, blend=1.0,
                     grip=plan.SHAKER_SQUEEZE, stroke=plan.STROKE,
                     tip=plan.TIP_ANGLE, grasping=False, carrying=True,
                     elapsed=25.0)
    )
    assert held.seasoned is True


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


def test_each_grasp_commands_its_own_squeeze() -> None:
    for phase in plan.PHASES:
        if phase.label in plan.GRASPING:
            # An 18 mm bar: the fingers are commanded shut and stall on it.
            assert phase.grip == plan.HANDLE_SQUEEZE
        elif phase.label in plan.CARRYING:
            # A 30 mm waist: closing all the way would stall the fingers deep
            # inside the glass and drive the actuators to their force limit.
            assert phase.grip == plan.SHAKER_SQUEEZE
        else:
            assert phase.grip == plan.OPEN


def test_the_seasoning_errand_happens_while_the_drawer_stands_open() -> None:
    labels = [p.label for p in plan.PHASES]
    # Nothing can be taken out of a shut drawer, so the whole excursion has to
    # fall between the pull and the push.
    pull, push = (labels.index("pull the drawer open"),
                  labels.index("push the drawer shut"))
    for label in plan.CARRYING:
        assert pull < labels.index(label) < push


def test_the_shaker_goes_back_before_the_drawer_is_shut() -> None:
    labels = [p.label for p in plan.PHASES]
    assert (labels.index("lower it back in")
            < labels.index(plan.SHAKER_RELEASED)
            < labels.index("push the drawer shut"))


def test_the_two_grasps_are_squared_up_differently() -> None:
    # The handle is pinched top to bottom and the shaker is taken from above,
    # so no phase may claim both poses at once, and every phase that carries
    # the shaker has to be off the handle's square.
    modes = {p.label: p.mode for p in plan.PHASES}
    for label in plan.GRASPING:
        assert modes[label] in plan.SQUARE_TO_HANDLE
    for label in plan.CARRYING:
        assert modes[label] not in plan.SQUARE_TO_HANDLE


def test_the_torso_rises_for_the_drawer_and_drops_for_the_plate() -> None:
    torso = {p.label: p.torso for p in plan.PHASES}
    # The Panda cannot cover a handle a metre up and a plate on the tabletop
    # from one shoulder height, so the Omron lift is driven between them.
    assert torso["pull the drawer open"] == plan.TORSO_HIGH
    assert torso["down to the plate"] == plan.TORSO_LOW
    assert torso["shake the seasoning"] == plan.TORSO_LOW
    assert torso["push the drawer shut"] == plan.TORSO_HIGH


def test_the_commanded_roll_only_leaves_upright_over_the_plate() -> None:
    # `tip_at` is what the mock judges the seasoning by, so it has to be zero
    # everywhere except the two tilts and the shake.
    tipping = {"tip it over the food", "shake the seasoning", "bring it upright"}
    for phase in plan.PHASES:
        for blend in (0.0, 0.5, 1.0):
            angle = plan.tip_at(phase, blend)
            if phase.label in tipping:
                assert 0.0 <= angle <= plan.TIP_ANGLE + plan.SHAKE_ROLL
            else:
                assert angle == 0.0
    # And it does actually go cap-down: past the threshold verify.py asserts.
    assert plan.tip_at(plan.PHASES[14], 1.0) >= plan.TIP_MIN


def test_mujoco_lane_explains_itself_when_the_scene_is_missing() -> None:
    simulator = DrawerSimulator(
        mode="mujoco", scene="/nonexistent/scene.xml", camera="action", hz=20.0
    )
    with pytest.raises(RuntimeError) as excinfo:
        simulator.reset()

    message = str(excinfo.value)
    assert "demo-drawer-mock" in message or "scene" in message


def test_the_mock_lane_never_reaches_for_the_viewer() -> None:
    # The viewer is an optional lane; importing the example must not pull it,
    # or the mock-safe path would inherit MuJoCo and viser.
    assert "examples.advanced.robocasa_drawer.viewer" not in sys.modules
    assert "examples.advanced.robocasa_drawer.scene" not in sys.modules
    assert "viser" not in sys.modules


class TestViewerGeometry:
    """Geometry helpers for the browser viewer.

    Skipped wherever the optional simulator dependencies are absent, which
    includes the default environment and CI.
    """

    @staticmethod
    def _viewer():
        pytest.importorskip("mujoco")
        pytest.importorskip("viser")
        from examples.advanced.robocasa_drawer import viewer

        return viewer

    def test_box_mesh_is_a_closed_twelve_triangle_box(self) -> None:
        viewer = self._viewer()
        verts, faces = viewer._box_mesh([0.5, 0.25, 0.1])

        assert verts.shape == (8, 3)
        assert faces.shape == (12, 3)
        # Every edge shared by exactly two triangles: the box has no holes.
        edges: dict[tuple[int, int], int] = {}
        for tri in faces:
            for a, b in ((0, 1), (1, 2), (2, 0)):
                key = tuple(sorted((int(tri[a]), int(tri[b]))))
                edges[key] = edges.get(key, 0) + 1
        assert set(edges.values()) == {2}

    def test_cylinder_mesh_closes_both_caps(self) -> None:
        viewer = self._viewer()
        segments = 24
        verts, faces = viewer._cylinder_mesh(0.03, 0.2, segments=segments)

        assert verts.shape == (2 * segments + 2, 3)
        assert faces.shape == (4 * segments, 3)
        assert verts[:, 2].min() == pytest.approx(-0.2)
        assert verts[:, 2].max() == pytest.approx(0.2)

    def test_unbounded_plane_still_gets_drawn(self) -> None:
        viewer = self._viewer()
        # A MuJoCo plane with size 0 is infinite; it has to become something
        # finite or the floor simply does not appear in the browser.
        verts, faces = viewer._plane_mesh([0.0, 0.0, 0.05])

        assert faces.shape == (2, 3)
        assert abs(verts[:, 0]).max() > 0.0
