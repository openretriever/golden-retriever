"""D1 multi-drawer search: belief, policies, termination and determinism.

Mock-safe by construction — nothing here imports MuJoCo, RoboSuite or RoboCasa,
so the suite runs in CI, which never installs the simulator.
"""

from __future__ import annotations

import json

import pytest

from examples.advanced.robocasa.search_app import run_episode
from examples.advanced.robocasa.search_belief import (
    BeliefTracker,
    InspectionResult,
    SearchBelief,
)
from examples.advanced.robocasa.search_policy import get_policy
from examples.advanced.robocasa.search_scenes import SCENES, get_scene
from examples.advanced.robocasa.search_world import MockSearchWorld

SCENE_ID = "three_drawer_stack"
ALL_SCENES = sorted(SCENES)
POLICIES = ["fixed-order", "nearest-first"]

# Every container is exactly equidistant here, so cost ordering has nothing to
# work with and the two policies must behave identically. The control case.
EQUIDISTANT_SCENE = "island_ring"
# Travel cost varies ~3x end to end, so cost ordering should pay off most.
SPREAD_SCENE = "galley_run"


@pytest.fixture
def scene():
    return get_scene(SCENE_ID)


# --- belief ---------------------------------------------------------------


def test_uniform_belief_sums_to_one(scene):
    belief = SearchBelief.uniform(scene.container_ids)
    assert pytest.approx(sum(belief.probabilities)) == 1.0
    assert len(set(belief.probabilities)) == 1


def test_empty_container_is_ruled_out_and_mass_renormalizes(scene):
    belief = SearchBelief.uniform(scene.container_ids)
    updated = belief.update(InspectionResult("drawer_top", found=False))

    assert updated.probability("drawer_top") == 0.0
    assert pytest.approx(sum(updated.probabilities)) == 1.0
    # The remaining three now split the mass evenly.
    assert pytest.approx(updated.probability("drawer_middle")) == 1 / 3
    assert "drawer_top" in updated.inspected


def test_finding_the_target_collapses_the_belief(scene):
    belief = SearchBelief.uniform(scene.container_ids)
    updated = belief.update(InspectionResult("drawer_middle", found=True))

    assert updated.found is True
    assert updated.found_in == "drawer_middle"
    assert updated.probability("drawer_middle") == 1.0
    assert sum(updated.probabilities) == 1.0


def test_belief_updates_never_mutate_in_place(scene):
    belief = SearchBelief.uniform(scene.container_ids)
    before = belief.probabilities
    belief.update(InspectionResult("drawer_top", found=False))
    assert belief.probabilities == before


def test_ruling_out_every_container_exhausts_the_belief(scene):
    belief = SearchBelief.uniform(scene.container_ids)
    for container_id in scene.container_ids:
        belief = belief.update(InspectionResult(container_id, found=False))

    assert belief.exhausted is True
    assert belief.candidates == ()
    with pytest.raises(ValueError):
        belief.most_likely()


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_belief_stays_a_valid_distribution_through_a_whole_search(scene_id):
    """The core invariant, on the belief itself rather than its rounded view.

    A belief that does not sum to 1 is not a probability distribution, and every
    downstream claim — candidates, collapse, the (n+1)/2 result — rests on it.
    """
    target = get_scene(scene_id)
    belief = SearchBelief.uniform(target.container_ids)
    assert sum(belief.probabilities) == pytest.approx(1.0)

    for container_id in target.container_ids[:-1]:
        belief = belief.update(InspectionResult(container_id, found=False))
        assert sum(belief.probabilities) == pytest.approx(1.0)
        assert all(0.0 <= p <= 1.0 for p in belief.probabilities)

    # Ruling out the last one leaves no mass anywhere, which is the one legal
    # departure from summing to 1 and is what `exhausted` reports.
    belief = belief.update(InspectionResult(target.container_ids[-1], found=False))
    assert belief.exhausted
    assert sum(belief.probabilities) == pytest.approx(0.0)


@pytest.mark.parametrize("scene_id", ALL_SCENES)
@pytest.mark.parametrize("policy_name", POLICIES)
def test_traced_belief_sums_to_one_within_its_rounding(scene_id, policy_name):
    """`as_dict()` rounds to 6 decimals for the trace, so 1/3 and 1/6 cannot sum
    to exactly 1. The tolerance here is that rounding, nothing more — a real
    normalization bug moves the sum far further than 1e-5."""
    for seed in range(10):
        trace = run_episode(scene_id=scene_id, seed=seed, policy_name=policy_name)
        for record in trace.records:
            for snapshot in (record.belief_before, record.belief_after):
                if not snapshot:
                    continue
                assert sum(snapshot.values()) == pytest.approx(1.0, abs=1e-5)
                assert all(0.0 <= p <= 1.0 for p in snapshot.values())


def test_belief_rejects_an_unknown_container(scene):
    belief = SearchBelief.uniform(scene.container_ids)
    with pytest.raises(KeyError):
        belief.update(InspectionResult("no_such_drawer", found=False))


def test_belief_tracker_flow_resets_between_episodes(scene):
    tracker = BeliefTracker(container_ids=scene.container_ids)
    tracker.reset()
    tracker.step(InspectionResult("drawer_top", found=False))
    assert tracker.belief.probability("drawer_top") == 0.0

    tracker.reset()
    assert pytest.approx(tracker.belief.probability("drawer_top")) == 0.25
    assert tracker.init_config()["container_ids"] == list(scene.container_ids)


# --- scenes ---------------------------------------------------------------


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_target_placement_is_deterministic(scene_id):
    target = get_scene(scene_id)
    assert target.target_container(7) == target.target_container(7)


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_target_moves_across_seeds(scene_id):
    target = get_scene(scene_id)
    placements = {target.target_container(seed) for seed in range(80)}
    # Every container must be reachable, or the search task is degenerate.
    assert placements == set(target.container_ids)


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_scenes_mix_drawers_and_cabinets(scene_id):
    """D1 asks for drawers *or cabinets*; a scene of one kind is a weaker test."""
    kinds = {c.kind for c in get_scene(scene_id).containers}
    assert kinds == {"drawer", "cabinet"}


def test_three_fixed_scenes_exist():
    assert len(SCENES) == 3


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_declaration_order_is_not_distance_order(scene_id):
    """If a scene listed containers nearest-first, the two policies collapse
    into one and the comparison would measure nothing."""
    target = get_scene(scene_id)
    if target.scene_id == EQUIDISTANT_SCENE:
        pytest.skip("equidistant by construction; every order is distance order")
    costs = [target.travel_cost(cid) for cid in target.container_ids]
    assert costs != sorted(costs)


def test_equidistant_scene_really_is_equidistant():
    target = get_scene(EQUIDISTANT_SCENE)
    costs = {round(target.travel_cost(cid), 9) for cid in target.container_ids}
    assert len(costs) == 1


def test_scene_rejects_an_unknown_container(scene):
    with pytest.raises(KeyError):
        scene.container("no_such_drawer")


def test_unknown_scene_names_the_known_ones():
    with pytest.raises(KeyError, match=SCENE_ID):
        get_scene("not_a_scene")


# --- policies -------------------------------------------------------------


@pytest.mark.parametrize("policy_name", POLICIES)
def test_policy_stops_once_the_target_is_found(policy_name, scene):
    policy = get_policy(policy_name)
    belief = SearchBelief.uniform(scene.container_ids).update(
        InspectionResult("drawer_top", found=True)
    )
    decision = policy.decide(belief, scene)

    assert decision.stops
    assert decision.reason == "target found"


@pytest.mark.parametrize("policy_name", POLICIES)
def test_policy_stops_when_every_container_is_ruled_out(policy_name, scene):
    policy = get_policy(policy_name)
    belief = SearchBelief.uniform(scene.container_ids)
    for container_id in scene.container_ids:
        belief = belief.update(InspectionResult(container_id, found=False))

    assert policy.decide(belief, scene).stops


@pytest.mark.parametrize("policy_name", POLICIES)
def test_policy_never_picks_a_ruled_out_container(policy_name, scene):
    policy = get_policy(policy_name)
    belief = SearchBelief.uniform(scene.container_ids).update(
        InspectionResult("drawer_top", found=False)
    )
    assert policy.decide(belief, scene).container_id != "drawer_top"


def test_fixed_order_follows_scene_declaration_order(scene):
    policy = get_policy("fixed-order")
    belief = SearchBelief.uniform(scene.container_ids)
    assert policy.decide(belief, scene).container_id == scene.container_ids[0]


def test_nearest_first_picks_the_cheapest_container(scene):
    policy = get_policy("nearest-first")
    belief = SearchBelief.uniform(scene.container_ids)
    chosen = policy.decide(belief, scene).container_id
    cheapest = min(scene.container_ids, key=scene.travel_cost)
    assert chosen == cheapest


def test_unknown_policy_names_the_known_ones():
    with pytest.raises(KeyError, match="nearest-first"):
        get_policy("not_a_policy")


# --- episodes -------------------------------------------------------------


@pytest.mark.parametrize("policy_name", POLICIES)
@pytest.mark.parametrize("seed", range(6))
def test_episode_always_finds_the_target(policy_name, seed):
    trace = run_episode(scene_id=SCENE_ID, seed=seed, policy_name=policy_name)
    assert trace.succeeded
    assert trace.records[-1].event == "terminated"


@pytest.mark.parametrize("policy_name", POLICIES)
@pytest.mark.parametrize("seed", range(6))
def test_episode_never_reinspects_a_container(policy_name, seed):
    trace = run_episode(scene_id=SCENE_ID, seed=seed, policy_name=policy_name)
    assert trace.repeated_inspections == 0
    assert trace.containers_opened == trace.steps


@pytest.mark.parametrize("policy_name", POLICIES)
@pytest.mark.parametrize("seed", range(6))
def test_episode_stops_at_the_target_not_after_it(policy_name, seed):
    """The last look must be the successful one — searching on would be waste."""
    trace = run_episode(scene_id=SCENE_ID, seed=seed, policy_name=policy_name)
    inspections = trace.inspections
    assert inspections[-1].found is True
    assert not any(record.found for record in inspections[:-1])


def test_episode_opens_no_more_containers_than_exist(scene):
    for seed in range(20):
        trace = run_episode(scene_id=SCENE_ID, seed=seed, policy_name="fixed-order")
        assert trace.containers_opened <= len(scene.containers)


def test_world_verification_agrees_with_the_trace():
    world = MockSearchWorld()
    trace = run_episode(scene_id=SCENE_ID, seed=3, policy_name="nearest-first", world=world)
    verification = world.verify()

    assert verification.success is trace.succeeded
    assert verification.target_container == get_scene(SCENE_ID).target_container(3)


def test_world_rejects_inspection_before_reset():
    with pytest.raises(RuntimeError):
        MockSearchWorld().inspect("drawer_top")


# --- determinism ----------------------------------------------------------


@pytest.mark.parametrize("policy_name", POLICIES)
def test_same_seed_replays_an_identical_trace(policy_name):
    """The D1 done-definition: fixed scenes reset and replay identically."""
    first = run_episode(scene_id=SCENE_ID, seed=11, policy_name=policy_name)
    second = run_episode(scene_id=SCENE_ID, seed=11, policy_name=policy_name)
    assert first.to_jsonl() == second.to_jsonl()


@pytest.mark.parametrize("scene_id", ALL_SCENES)
@pytest.mark.parametrize("policy_name", POLICIES)
def test_every_scene_is_solvable_by_every_policy(scene_id, policy_name):
    for seed in range(12):
        trace = run_episode(scene_id=scene_id, seed=seed, policy_name=policy_name)
        assert trace.succeeded
        assert trace.repeated_inspections == 0


# --- what the policies actually differ on ---------------------------------


def test_ordering_does_not_change_how_many_containers_are_opened():
    """With a uniformly placed target, expected containers opened is (n+1)/2
    whatever order you use. A policy claiming to improve this is reporting
    noise, so the metric must not be used to rank policies."""
    seeds = range(200)
    means = {}
    for policy_name in POLICIES:
        opened = [
            run_episode(scene_id=SPREAD_SCENE, seed=s, policy_name=policy_name).containers_opened
            for s in seeds
        ]
        means[policy_name] = sum(opened) / len(opened)

    scene_size = len(get_scene(SPREAD_SCENE).containers)
    expected = (scene_size + 1) / 2
    for mean in means.values():
        assert mean == pytest.approx(expected, abs=0.5)


def test_nearest_first_walks_less_where_geometry_varies():
    """Travel cost is the metric ordering actually moves."""
    seeds = range(200)
    totals = {
        policy_name: sum(
            run_episode(scene_id=SPREAD_SCENE, seed=s, policy_name=policy_name).total_travel_cost
            for s in seeds
        )
        for policy_name in POLICIES
    }
    assert totals["nearest-first"] < totals["fixed-order"] * 0.75


def test_policies_are_indistinguishable_on_an_equidistant_scene():
    """The control: with nothing to order by, the two must agree exactly."""
    for seed in range(20):
        fixed = run_episode(scene_id=EQUIDISTANT_SCENE, seed=seed, policy_name="fixed-order")
        nearest = run_episode(scene_id=EQUIDISTANT_SCENE, seed=seed, policy_name="nearest-first")
        assert fixed.containers_opened == nearest.containers_opened
        assert fixed.total_travel_cost == pytest.approx(nearest.total_travel_cost)


def test_travel_cost_is_the_sum_of_its_legs():
    trace = run_episode(scene_id=SPREAD_SCENE, seed=4, policy_name="nearest-first")
    assert trace.total_travel_cost == pytest.approx(
        sum(r.travel_cost for r in trace.inspections)
    )
    assert all(r.travel_cost >= 0.0 for r in trace.inspections)


def test_first_leg_starts_from_the_robot_home():
    scene_obj = get_scene(SPREAD_SCENE)
    trace = run_episode(scene_id=SPREAD_SCENE, seed=2, policy_name="nearest-first")
    first = trace.inspections[0]
    assert first.travel_cost == pytest.approx(scene_obj.travel_cost(first.container_id))


def test_different_seeds_eventually_diverge():
    traces = {
        run_episode(scene_id=SCENE_ID, seed=seed, policy_name="nearest-first").to_jsonl()
        for seed in range(12)
    }
    assert len(traces) > 1


# --- trace ----------------------------------------------------------------


def test_trace_is_valid_jsonl_and_self_describing(tmp_path):
    trace = run_episode(scene_id=SCENE_ID, seed=5, policy_name="fixed-order")
    written = trace.write(tmp_path / "trace.jsonl")
    lines = written.read_text(encoding="utf-8").strip().splitlines()

    header = json.loads(lines[0])
    assert header["scene_id"] == SCENE_ID
    assert header["seed"] == 5
    assert header["policy"] == "fixed-order"

    records = [json.loads(line) for line in lines[1:]]
    assert records[0]["event"] == "episode_started"
    assert records[-1]["event"] == "terminated"
    # Every inspection records the belief on both sides of the update, which is
    # what makes the update auditable rather than asserted.
    for record in records:
        if record["event"] == "inspected":
            assert record["belief_before"] and record["belief_after"]
            assert record["reason"]


def test_trace_render_shows_the_search_story():
    trace = run_episode(scene_id=SCENE_ID, seed=1, policy_name="nearest-first")
    rendered = trace.render()

    assert SCENE_ID in rendered
    assert "FOUND" in rendered
    assert "success=True" in rendered
    for record in trace.inspections:
        assert record.container_id in rendered


def test_scene_registry_is_not_empty():
    assert SCENES
