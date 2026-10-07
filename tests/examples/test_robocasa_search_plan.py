"""The plan in the trace, and the replan counter.

The D1 milestone asks the trace to show "plan, opened containers, observations,
belief update, and termination". This covers the plan half, and proves the
replan counter can actually fire — an instrument that only ever reads zero is
indistinguishable from one that is broken.

Mock-safe: no simulator import, so this gates CI.
"""

from __future__ import annotations

import json

import pytest

from examples.advanced.robocasa.search_app import run_episode
from examples.advanced.robocasa.search_belief import SearchBelief
from examples.advanced.robocasa.search_metrics import EpisodeMetrics
from examples.advanced.robocasa.search_plan import SearchPlan, plan_ahead
from examples.advanced.robocasa.search_policy import SearchDecision, get_policy
from examples.advanced.robocasa.search_scenes import SCENES, get_scene

ALL_SCENES = sorted(SCENES)
POLICIES = ["fixed-order", "nearest-first"]
SCENE_ID = "galley_run"


# --- lookahead ------------------------------------------------------------


@pytest.mark.parametrize("policy_name", POLICIES)
def test_plan_covers_every_container_from_a_uniform_belief(policy_name):
    scene = get_scene(SCENE_ID)
    plan = plan_ahead(
        get_policy(policy_name), SearchBelief.uniform(scene.container_ids), scene
    )
    assert set(plan.container_ids) == set(scene.container_ids)
    assert len(plan.container_ids) == len(scene.containers)


@pytest.mark.parametrize("policy_name", POLICIES)
def test_plan_omits_containers_already_ruled_out(policy_name):
    from examples.advanced.robocasa.search_belief import InspectionResult

    scene = get_scene(SCENE_ID)
    belief = SearchBelief.uniform(scene.container_ids).update(
        InspectionResult("drawer_near", found=False)
    )
    plan = plan_ahead(get_policy(policy_name), belief, scene)
    assert "drawer_near" not in plan.container_ids


def test_plan_is_empty_once_the_target_is_found():
    from examples.advanced.robocasa.search_belief import InspectionResult

    scene = get_scene(SCENE_ID)
    belief = SearchBelief.uniform(scene.container_ids).update(
        InspectionResult("drawer_near", found=True)
    )
    assert plan_ahead(get_policy("nearest-first"), belief, scene).container_ids == ()


def test_fixed_order_plan_is_scene_order():
    scene = get_scene(SCENE_ID)
    plan = plan_ahead(
        get_policy("fixed-order"), SearchBelief.uniform(scene.container_ids), scene
    )
    assert plan.container_ids == scene.container_ids


def test_nearest_first_plan_is_a_route_not_a_distance_sort():
    """It re-measures from each stop, so the plan is a greedy tour rather than
    containers sorted by distance from home."""
    scene = get_scene(SCENE_ID)
    plan = plan_ahead(
        get_policy("nearest-first"), SearchBelief.uniform(scene.container_ids), scene
    )
    by_distance_from_home = tuple(sorted(scene.container_ids, key=scene.travel_cost))
    assert plan.container_ids != by_distance_from_home


def test_lookahead_does_not_mutate_the_belief():
    scene = get_scene(SCENE_ID)
    belief = SearchBelief.uniform(scene.container_ids)
    before = belief.probabilities
    plan_ahead(get_policy("nearest-first"), belief, scene)
    assert belief.probabilities == before


# --- the plan reaches the trace ------------------------------------------


@pytest.mark.parametrize("scene_id", ALL_SCENES)
@pytest.mark.parametrize("policy_name", POLICIES)
def test_trace_opens_with_a_plan(scene_id, policy_name):
    trace = run_episode(scene_id=scene_id, seed=3, policy_name=policy_name)
    assert trace.plans, "no plan was recorded"
    assert trace.initial_plan
    assert trace.plans[0].plan["policy"] == policy_name


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_trace_carries_all_five_required_elements(scene_id):
    """plan, opened containers, observations, belief update, termination."""
    trace = run_episode(scene_id=scene_id, seed=2, policy_name="nearest-first")
    events = {r.event for r in trace.records}
    scene = get_scene(scene_id)

    # plan
    assert "planned" in events
    assert trace.initial_plan

    # opened containers — each one named, and a real container in this scene
    assert "inspected" in events
    assert all(r.container_id in scene.container_ids for r in trace.inspections)

    # observations — exactly one inspection found the target, and it is the last
    found = [r for r in trace.inspections if r.found]
    assert len(found) == 1
    assert found[0] is trace.inspections[-1]

    # belief update — every inspection moved the distribution, and both sides
    # of the update are on the record
    for record in trace.inspections:
        assert record.belief_before and record.belief_after
        assert record.belief_before != record.belief_after
        assert record.belief_after[record.container_id] in (0.0, 1.0)

    # termination
    assert "terminated" in events
    assert trace.records[-1].event == "terminated"
    assert trace.status == "SUCCESS"


def test_plan_survives_the_trace_round_trip(tmp_path):
    trace = run_episode(scene_id=SCENE_ID, seed=7, policy_name="nearest-first")
    path = trace.write(tmp_path / "trace.jsonl")
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    planned = [json.loads(line) for line in lines if '"planned"' in line]

    assert planned
    assert planned[0]["plan"]["container_ids"] == trace.initial_plan
    assert EpisodeMetrics.from_jsonl(path).replans == trace.replans


# --- replans -------------------------------------------------------------


@pytest.mark.parametrize("scene_id", ALL_SCENES)
@pytest.mark.parametrize("policy_name", POLICIES)
def test_deterministic_policies_never_replan(scene_id, policy_name):
    """With noiseless observations and a policy that depends only on belief and
    position, the lookahead is exact — so zero replans is the correct answer,
    not a broken counter. `test_a_stateful_policy_forces_a_replan` proves the
    counter still works."""
    for seed in range(12):
        trace = run_episode(scene_id=scene_id, seed=seed, policy_name=policy_name)
        assert trace.replans == 0
        assert trace.followed_initial_plan


class _DriftingPolicy:
    """A policy with internal state, so lookahead cannot predict it.

    Realistic rather than contrived: any policy carrying its own memory breaks
    the pure-function assumption `plan_ahead` relies on, and that is exactly
    when a replan should be recorded.
    """

    name = "drifting"

    def __init__(self) -> None:
        self.calls = 0

    def decide(self, belief, scene, position=None) -> SearchDecision:
        if belief.found:
            return SearchDecision("stop", belief.found_in, "target found")
        candidates = [c for c in scene.container_ids if c in belief.candidates]
        if not candidates:
            return SearchDecision("stop", "", "every container ruled out")
        self.calls += 1
        # The choice rotates with call count, which lookahead cannot foresee.
        return SearchDecision(
            "inspect", candidates[self.calls % len(candidates)], "rotating choice"
        )


def _drifting_traces(monkeypatch, seeds=range(12)):
    """Run the drifting policy across seeds.

    A replan can only be detected after the first inspection, so an episode
    that finds the target immediately cannot show one. Scan seeds rather than
    betting on a single one.
    """
    from examples.advanced.robocasa import search_app as app

    monkeypatch.setattr(app, "get_policy", lambda name: _DriftingPolicy())
    return [
        run_episode(scene_id=SCENE_ID, seed=seed, policy_name="drifting")
        for seed in seeds
    ]


def test_a_stateful_policy_forces_a_replan(monkeypatch):
    traces = _drifting_traces(monkeypatch)
    replanned = [t for t in traces if t.replans > 0]

    assert replanned, "the replan counter never fired on any seed"
    for trace in traces:
        assert len(trace.plans) == trace.replans + 1
        assert trace.succeeded


def test_replans_reach_the_metrics(monkeypatch):
    traces = _drifting_traces(monkeypatch)
    pairs = [(EpisodeMetrics.from_trace(t), t) for t in traces]

    assert all(m.replans == t.replans for m, t in pairs)
    assert any(m.replans > 0 for m, _ in pairs)


def test_a_replanning_episode_reports_not_following_its_initial_plan(monkeypatch):
    traces = _drifting_traces(monkeypatch)
    diverged = [t for t in traces if t.replans > 0]
    assert any(not t.followed_initial_plan for t in diverged)


def test_plan_payload_round_trips_through_its_dict():
    plan = SearchPlan(policy="p", revision=2, container_ids=("a", "b"), basis="why")
    assert plan.as_dict() == {
        "policy": "p",
        "revision": 2,
        "container_ids": ["a", "b"],
        "basis": "why",
    }
    assert plan.next_container == "a"
    assert SearchPlan().next_container == ""
