"""D1 search against a real RoboCasa kitchen.

Opt-in: these need RoboCasa, RoboSuite and ~10 GB of kitchen assets, so they
are skipped unless `RUN_ROBOCASA_SEARCH_TESTS=1`, matching how
`test_robocasa_replay_data.py` gates its own real-asset checks. CI never
installs the simulator; `test_robocasa_search.py` covers the same logic
mock-first and does gate CI.

    pixi run --locked -e robocasa test-robocasa-search-sim
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_ROBOCASA_SEARCH_TESTS") != "1",
    reason="set RUN_ROBOCASA_SEARCH_TESTS=1 and install the robocasa environment",
)

LAYOUT = 5
CONTAINERS = 5


@pytest.fixture(scope="module")
def world():
    """One kitchen for the whole module — building one costs far more than a
    search does, and `reset()` restores it between episodes."""
    from examples.advanced.robocasa.search_robocasa_world import RoboCasaSearchWorld

    built = RoboCasaSearchWorld(layout_id=LAYOUT, max_containers=CONTAINERS)
    try:
        yield built
    finally:
        built.close()


@pytest.fixture(scope="module")
def scene(world):
    return world.build_scene(scene_id=f"robocasa_layout{LAYOUT}")


def _run(scene, world, seed, policy="nearest-first"):
    from examples.advanced.robocasa.search_app import run_episode

    return run_episode(
        scene=scene, seed=seed, policy_name=policy, world=world, close_world=False
    )


def test_scene_is_built_from_real_fixtures(scene):
    assert len(scene.containers) == CONTAINERS
    # Real drawers, not registry names.
    assert all(c.container_id.startswith("stack_") for c in scene.containers)
    assert all(c.kind == "drawer" for c in scene.containers)


def test_containers_are_spread_across_the_kitchen(scene):
    """One drawer per stack. If selection collapsed onto a single stack every
    cost would land within a few centimetres and ordering would measure
    nothing."""
    costs = [scene.travel_cost(c) for c in scene.container_ids]
    assert max(costs) / min(costs) > 1.5


def test_declaration_order_is_not_distance_order(scene):
    """Otherwise `fixed-order` silently becomes `nearest-first`."""
    costs = [scene.travel_cost(c) for c in scene.container_ids]
    assert costs != sorted(costs)


def test_robot_home_is_not_the_parked_dummy_body(scene):
    """`robot0_base` sits at (10, 10, 0) on a mobile-base robot; reading it
    would put the robot 14 m from its own kitchen."""
    assert max(abs(v) for v in scene.robot_home) < 9.0


@pytest.mark.parametrize("seed", range(4))
def test_search_finds_the_target_in_real_physics(scene, world, seed):
    trace = _run(scene, world, seed)
    assert trace.succeeded
    assert trace.repeated_inspections == 0
    assert trace.records[-1].event == "terminated"


@pytest.mark.parametrize("seed", range(4))
def test_world_verification_agrees_with_the_agent(scene, world, seed):
    """The world's own geometry decides success, not the agent's belief."""
    trace = _run(scene, world, seed)
    verification = world.verify()
    assert verification.success is trace.succeeded
    assert verification.target_container == scene.target_container(seed)


def test_only_the_container_holding_the_target_reports_found(scene, world):
    trace = _run(scene, world, 2)
    found = [r for r in trace.inspections if r.found]
    assert len(found) == 1
    assert found[0].container_id == scene.target_container(2)


def test_same_seed_replays_an_identical_trace(scene, world):
    """The D1 done-definition, against real physics rather than the mock."""
    first = _run(scene, world, 5)
    second = _run(scene, world, 5)
    assert first.to_jsonl() == second.to_jsonl()


def test_target_moves_across_seeds(scene):
    placements = {scene.target_container(seed) for seed in range(60)}
    assert placements == set(scene.container_ids)


def test_mock_and_real_agree_on_where_the_target_hides(scene):
    """Placement is computed in `scenes.py`, never by RoboCasa's RNG, so the
    two backends must agree — that is what makes the mock lane trustworthy."""
    from examples.advanced.robocasa.search_world import MockSearchWorld

    mock = MockSearchWorld()
    for seed in range(8):
        mock.reset(scene, seed)
        assert mock.verify().target_container == scene.target_container(seed)
