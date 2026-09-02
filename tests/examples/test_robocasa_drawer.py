"""Mock-lane tests for the RoboCasa drawer example.

These run in the default environment, which has neither MuJoCo nor RoboCasa
installed, so they double as the guard that the example stays import-safe.
"""

import sys
from argparse import Namespace

import pytest
from examples.advanced.robocasa_drawer import plan
from examples.advanced.robocasa_drawer.app import (
    ChoreographyPolicy,
    DrawerAction,
    DrawerSimulator,
    build_pipeline,
)


def _mock_args(**overrides: object) -> Namespace:
    values = {
        "mode": "mock",
        "scene": None,
        "camera": "threequarter",
        "steps": 94,
        "dt": 0.5,
        "hz": 2.0,
        "print_every": 100,
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
    simulator = DrawerSimulator(mode="mock", scene=None, camera="threequarter", hz=2.0)
    simulator.reset()

    opened = simulator.step(
        DrawerAction(phase="pull drawer open", phase_index=4, blend=1.0,
                         grip=plan.SQUEEZE, stroke=plan.STROKE, grasping=True,
                         holding_jar=False, elapsed=8.0)
    )
    assert opened.drawer_open == pytest.approx(plan.STROKE)

    # Same commanded stroke, but nothing is holding the bar.
    released = simulator.step(
        DrawerAction(phase="line up", phase_index=1, blend=1.0,
                         grip=plan.OPEN, stroke=plan.STROKE, grasping=False,
                         holding_jar=False, elapsed=9.0)
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
        elif phase.label in plan.HOLDING_JAR:
            # The jar is a 48 mm cylinder, not an 18 mm bar: closing all the
            # way would stall the fingers deep inside it.
            assert phase.grip == plan.JAR_SQUEEZE
        else:
            assert phase.grip == plan.OPEN


def test_the_errand_happens_while_the_drawer_stands_open() -> None:
    labels = [p.label for p in plan.PHASES]
    # Nothing can be put into a shut drawer, so the whole jar errand has to
    # fall between the pull and the push.
    pull, push = labels.index("pull drawer open"), labels.index("push drawer shut")
    for label in plan.HOLDING_JAR:
        assert pull < labels.index(label) < push


def test_the_wrist_returns_to_neutral_between_the_two_grasps() -> None:
    # The handle and the jar want wrists a quarter turn apart, and the arm
    # will not make that turn at full stretch. Every switch between them has
    # to pass through a transit phase.
    labels = [p.label for p in plan.PHASES]
    modes = {p.label: p.mode for p in plan.PHASES}
    first_jar = min(labels.index(x) for x in plan.HOLDING_JAR)
    last_jar = max(labels.index(x) for x in plan.HOLDING_JAR)
    before = {modes[l] for l in labels[labels.index("back off"):first_jar]}
    after = {modes[l] for l in labels[last_jar:labels.index("back to handle")]}
    assert "transit_front" in before
    assert "transit_front" in after


def test_the_torso_rises_for_the_worktop_and_drops_for_the_drawer() -> None:
    torso = {p.label: p.torso for p in plan.PHASES}
    # The jar is fetched from a worktop the arm cannot reach with the torso
    # down, and put into a drawer it cannot reach with the torso up.
    assert torso["down to jar"] == plan.TORSO_UP
    assert torso["grip jar"] == plan.TORSO_UP
    assert torso["lower into drawer"] == plan.TORSO_DOWN
    assert torso["pull drawer open"] == plan.TORSO_DOWN


def test_mujoco_lane_explains_itself_when_the_scene_is_missing() -> None:
    simulator = DrawerSimulator(
        mode="mujoco", scene="/nonexistent/scene.xml", camera="threequarter", hz=20.0
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
