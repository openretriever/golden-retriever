"""The observation / action / reset / success / progress / failure interfaces.

The D1 milestone asks for all six to be exposed. The happy path never exercises
failure — the target is always in some container — so the failure cases here are
driven by deliberately broken worlds.

Mock-safe: no simulator import, so this gates CI.
"""

from __future__ import annotations

import pytest

from examples.advanced.robocasa.search_app import run_episode
from examples.advanced.robocasa.search_belief import InspectionResult
from examples.advanced.robocasa.search_metrics import EpisodeMetrics
from examples.advanced.robocasa.search_scenes import get_scene
from examples.advanced.robocasa.search_world import (
    MockSearchWorld,
    SearchWorld,
    search_status,
)

SCENE_ID = "three_drawer_stack"


# --- the six interfaces ---------------------------------------------------


def test_mock_world_satisfies_the_search_world_protocol():
    assert isinstance(MockSearchWorld(), SearchWorld)


def test_reset_observation_and_status_are_all_reachable():
    scene = get_scene(SCENE_ID)
    world = MockSearchWorld()

    observation = world.reset(scene, seed=0)          # reset + observation
    assert observation.scene_id == scene.scene_id
    assert observation.opened == ()

    status = world.status()                            # progress
    assert status.status == "IN_PROGRESS"
    assert status.progress == 0.0

    result = world.inspect(scene.container_ids[0])      # action
    assert isinstance(result, InspectionResult)

    assert world.verify().target_container              # success
    world.close()


def test_status_requires_a_reset_first():
    with pytest.raises(RuntimeError):
        MockSearchWorld().status()


# --- progress semantics ---------------------------------------------------


def test_progress_rises_as_containers_are_ruled_out():
    scene = get_scene(SCENE_ID)
    world = MockSearchWorld()
    world.reset(scene, seed=0)

    seen = [world.status().progress]
    for container_id in scene.container_ids:
        world.inspect(container_id)
        seen.append(world.status().progress)

    assert seen[0] == 0.0
    assert seen == sorted(seen), f"progress went backwards: {seen}"
    assert seen[-1] == 1.0


def test_progress_reaches_one_the_moment_the_target_is_found():
    scene = get_scene(SCENE_ID)
    world = MockSearchWorld()
    world.reset(scene, seed=0)
    world.inspect(scene.target_container(0))

    status = world.status()
    assert status.status == "SUCCESS"
    assert status.progress == 1.0


def test_trace_records_monotonic_progress():
    trace = run_episode(scene_id="galley_run", seed=4, policy_name="nearest-first")
    progress = [r.progress for r in trace.records]
    assert progress == sorted(progress)
    assert trace.final_progress == 1.0


def test_search_status_rejects_an_empty_container_set():
    with pytest.raises(ValueError):
        search_status(inspected=0, total=0, found=False)


# --- failure: the target is nowhere --------------------------------------


class _AlwaysEmptyWorld:
    """A world where the target is in no container at all.

    Not reachable through normal placement — the point is to drive the
    exhaustion branch, which the happy path can never reach.
    """

    def __init__(self) -> None:
        self.scene = None
        self._opened: list[str] = []

    def reset(self, scene, seed):
        self.scene = scene
        self._opened = []
        return None

    def inspect(self, container_id):
        self._opened.append(container_id)
        return InspectionResult(container_id=container_id, found=False, step=len(self._opened))

    def observe(self):
        return None

    def status(self):
        return search_status(
            inspected=len(self._opened), total=len(self.scene.containers), found=False
        )

    def verify(self):
        return None

    def close(self) -> None:
        return None


def test_exhausting_every_container_is_a_failure_not_a_hang():
    trace = run_episode(scene_id=SCENE_ID, seed=0, world=_AlwaysEmptyWorld())

    assert trace.succeeded is False
    assert trace.failed is True
    assert trace.status == "FAILURE"
    assert trace.containers_opened == len(get_scene(SCENE_ID).containers)
    assert "not found" in trace.error_message
    assert trace.final_progress == 1.0


def test_failure_is_visible_in_the_metrics():
    trace = run_episode(scene_id=SCENE_ID, seed=0, world=_AlwaysEmptyWorld())
    metrics = EpisodeMetrics.from_trace(trace)

    assert metrics.failed is True
    assert metrics.status == "FAILURE"
    assert metrics.success is False
    assert metrics.error_message


def test_failure_survives_the_trace_round_trip(tmp_path):
    trace = run_episode(scene_id=SCENE_ID, seed=0, world=_AlwaysEmptyWorld())
    path = trace.write(tmp_path / "failed.jsonl")
    recovered = EpisodeMetrics.from_jsonl(path)

    assert recovered.status == "FAILURE"
    assert recovered.error_message == trace.error_message


# --- failure: the world itself breaks ------------------------------------


class _BrokenWorld(_AlwaysEmptyWorld):
    """Raises on the first inspection, like a simulator losing its model."""

    def inspect(self, container_id):
        raise RuntimeError("simulator lost the drawer")


def test_a_world_error_is_recorded_rather_than_crashing_the_episode():
    trace = run_episode(scene_id=SCENE_ID, seed=0, world=_BrokenWorld())

    assert trace.failed is True
    assert trace.records[-1].event == "failed"
    assert "simulator lost the drawer" in trace.error_message
    # The decision that led to the failure is still on the record.
    assert trace.records[-1].container_id
    assert trace.records[-1].reason


def test_a_world_error_still_produces_a_readable_trace():
    trace = run_episode(scene_id=SCENE_ID, seed=0, world=_BrokenWorld())
    rendered = trace.render()
    assert "FAILED" in rendered
    assert "status=FAILURE" in rendered


# --- the happy path still reports SUCCESS --------------------------------


@pytest.mark.parametrize("policy_name", ["fixed-order", "nearest-first"])
def test_successful_episodes_report_success(policy_name):
    trace = run_episode(scene_id=SCENE_ID, seed=3, policy_name=policy_name)
    assert trace.status == "SUCCESS"
    assert trace.failed is False
    assert trace.error_message == ""
