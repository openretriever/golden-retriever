"""D1 metrics and determinism.

Mock-safe: no simulator import anywhere, so this gates CI alongside
`test_robocasa_search.py`.
"""

from __future__ import annotations

import pytest

from examples.advanced.robocasa.search_app import run_episode
from examples.advanced.robocasa.search_metrics import (
    AggregateMetrics,
    EpisodeMetrics,
    render_table,
    replay_is_deterministic,
    sweep,
)
from examples.advanced.robocasa.search_policy import POLICIES
from examples.advanced.robocasa.search_scenes import SCENES, get_scene

ALL_SCENES = sorted(SCENES)
ALL_POLICIES = sorted(POLICIES)
SPREAD_SCENE = "galley_run"
EQUIDISTANT_SCENE = "island_ring"


# --- metrics come from the trace, not from internals ----------------------


def test_metrics_match_the_trace_they_came_from():
    trace = run_episode(scene_id=SPREAD_SCENE, seed=9, policy_name="nearest-first")
    metrics = EpisodeMetrics.from_trace(trace)

    assert metrics.success is trace.succeeded
    assert metrics.containers_opened == trace.containers_opened
    assert metrics.steps == trace.steps
    assert metrics.travel_cost == pytest.approx(trace.total_travel_cost)


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_metrics_survive_a_round_trip_through_a_trace_file(scene_id, tmp_path):
    """The done-definition: a trace is inspectable on its own. If the numbers
    recomputed from disk differ, the trace is not self-sufficient."""
    trace = run_episode(scene_id=scene_id, seed=6, policy_name="nearest-first")
    in_memory = EpisodeMetrics.from_trace(trace)
    path = trace.write(tmp_path / "trace.jsonl")
    from_disk = EpisodeMetrics.from_jsonl(path)

    assert from_disk.scene_id == in_memory.scene_id
    assert from_disk.seed == in_memory.seed
    assert from_disk.policy == in_memory.policy
    assert from_disk.success == in_memory.success
    assert from_disk.containers_opened == in_memory.containers_opened
    assert from_disk.repeated_inspections == in_memory.repeated_inspections
    assert from_disk.unnecessary_inspections == in_memory.unnecessary_inspections
    assert from_disk.steps == in_memory.steps
    assert from_disk.travel_cost == pytest.approx(in_memory.travel_cost)


def test_elapsed_time_is_not_written_into_the_trace():
    """Wall clock in the artifact would break byte-identical replay."""
    trace = run_episode(scene_id=SPREAD_SCENE, seed=1, policy_name="fixed-order")
    assert "elapsed" not in trace.to_jsonl()


# --- determinism ----------------------------------------------------------


@pytest.mark.parametrize("scene_id", ALL_SCENES)
@pytest.mark.parametrize("policy_name", ALL_POLICIES)
def test_replay_is_deterministic(scene_id, policy_name):
    assert replay_is_deterministic(scene_id=scene_id, seed=11, policy_name=policy_name)


@pytest.mark.parametrize("scene_id", ALL_SCENES)
def test_determinism_holds_across_many_seeds(scene_id):
    for seed in range(8):
        assert replay_is_deterministic(
            scene_id=scene_id, seed=seed, policy_name="nearest-first", runs=3
        )


# --- aggregation ----------------------------------------------------------


def test_sweep_covers_every_combination():
    seeds = range(5)
    episodes = sweep(
        scene_ids=ALL_SCENES, policy_names=ALL_POLICIES, seeds=seeds
    )
    assert len(episodes) == len(ALL_SCENES) * len(ALL_POLICIES) * len(seeds)
    assert all(e.success for e in episodes)


def test_aggregate_reports_full_success_and_no_repeats():
    episodes = sweep(
        scene_ids=[SPREAD_SCENE], policy_names=["nearest-first"], seeds=range(30)
    )
    aggregate = AggregateMetrics.over(episodes)

    assert aggregate.episodes == 30
    assert aggregate.success_rate == 1.0
    assert aggregate.mean_repeated == 0.0
    assert aggregate.mean_travel > 0.0


def test_aggregate_rejects_an_empty_group():
    with pytest.raises(ValueError):
        AggregateMetrics.over([])


def test_render_table_has_one_row_per_scene_and_policy():
    episodes = sweep(scene_ids=ALL_SCENES, policy_names=ALL_POLICIES, seeds=range(4))
    table = render_table(episodes)
    lines = [line for line in table.splitlines() if line and not line.startswith("-")]

    assert len(lines) == 1 + len(ALL_SCENES) * len(ALL_POLICIES)  # header + rows
    for scene_id in ALL_SCENES:
        assert scene_id in table


# --- what the metrics say about the policies ------------------------------


def test_travel_separates_policies_but_container_count_does_not():
    """The headline finding, pinned so a future change cannot quietly invert it."""
    seeds = range(200)
    results = {
        policy_name: AggregateMetrics.over(
            sweep(scene_ids=[SPREAD_SCENE], policy_names=[policy_name], seeds=seeds)
        )
        for policy_name in ALL_POLICIES
    }

    expected_opened = (len(get_scene(SPREAD_SCENE).containers) + 1) / 2
    for aggregate in results.values():
        assert aggregate.mean_containers_opened == pytest.approx(expected_opened, abs=0.5)

    assert results["nearest-first"].mean_travel < results["fixed-order"].mean_travel * 0.75


def test_equidistant_scene_shows_no_difference():
    """The control, expressed in metrics rather than raw traces."""
    results = {
        policy_name: AggregateMetrics.over(
            sweep(
                scene_ids=[EQUIDISTANT_SCENE],
                policy_names=[policy_name],
                seeds=range(60),
            )
        )
        for policy_name in ALL_POLICIES
    }
    assert results["nearest-first"].mean_travel == pytest.approx(
        results["fixed-order"].mean_travel
    )


def test_no_episode_ever_repeats_an_inspection():
    episodes = sweep(scene_ids=ALL_SCENES, policy_names=ALL_POLICIES, seeds=range(40))
    assert all(e.repeated_inspections == 0 for e in episodes)


def test_unnecessary_inspections_only_happen_once_per_episode_at_most():
    """It fires only when the target sits in the last-searched container, so a
    single episode can never accrue more than one."""
    episodes = sweep(scene_ids=ALL_SCENES, policy_names=ALL_POLICIES, seeds=range(60))
    assert all(e.unnecessary_inspections <= 1 for e in episodes)
    assert any(e.unnecessary_inspections == 1 for e in episodes)
