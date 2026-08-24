from __future__ import annotations

import sys
from contextlib import nullcontext
from threading import Event, Thread
from types import SimpleNamespace

import numpy as np
import pytest

from examples.advanced.robocasa.embodied import EmbodiedGoal, OfflineEmbodiedPlanner
from examples.advanced.robocasa.mjviser_bridge import (
    _DEFAULT_CAMERA_PRESETS,
    MjviserBridge,
    ReplayControls,
    _apply_camera_preset,
    _camera_presets_from_robot,
    _events_html,
    _graph_html,
    _plan_html,
    _robot_tracking_body_id,
)


def test_replay_controls_pause_step_restart_and_speed() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=2)
    controls.set_total_steps(749)

    assert controls.claim_next_action() == (True, False)

    controls.set_paused(True)
    assert controls.claim_next_action() == (False, False)

    controls.request_step()
    assert controls.claim_next_action() == (True, False)
    assert controls.claim_next_action() == (False, False)

    controls.set_speed(0.5)
    controls.update_observation(
        episode_step=17,
        cycle=0,
        progress=18 / 749,
        reward=0.25,
        success=False,
        action_norm=1.5,
    )
    snapshot = controls.snapshot()
    assert snapshot.status == "Paused"
    assert snapshot.speed == 0.5
    assert snapshot.episode_step == 17
    assert snapshot.total_steps == 749

    controls.request_restart()
    assert controls.claim_next_action() == (True, True)
    snapshot = controls.snapshot()
    assert snapshot.status == "Restarting"
    assert snapshot.paused is False
    assert snapshot.progress == 0.0
    assert snapshot.cycle == 1

    with pytest.raises(ValueError, match="positive"):
        controls.set_speed(0.0)


def test_restart_ignores_late_observations_from_previous_cycle() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    controls.set_total_steps(10)
    controls.update_observation(
        episode_step=9,
        cycle=0,
        progress=1.0,
        reward=1.0,
        success=True,
        action_norm=0.5,
    )

    controls.request_restart()
    controls.update_observation(
        episode_step=9,
        cycle=0,
        progress=1.0,
        reward=1.0,
        success=True,
        action_norm=0.5,
    )

    snapshot = controls.snapshot()
    assert snapshot.cycle == 1
    assert snapshot.status == "Restarting"
    assert snapshot.success is False
    assert snapshot.progress == 0.0

    assert controls.claim_next_action() == (True, True)
    controls.update_observation(
        episode_step=0,
        cycle=1,
        progress=0.0,
        reward=0.0,
        success=False,
        action_norm=0.25,
    )
    assert controls.snapshot().status == "Running"


def test_repeated_restart_requests_advance_one_cycle_each() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)

    controls.request_restart()
    controls.request_restart()
    assert controls.snapshot().cycle == 1
    assert controls.claim_next_action() == (True, True)

    controls.update_observation(
        episode_step=0,
        cycle=1,
        progress=0.0,
        reward=0.0,
        success=False,
        action_norm=0.25,
    )
    controls.request_restart()
    assert controls.snapshot().cycle == 2
    assert controls.claim_next_action() == (True, True)


def test_live_graph_html_contains_typed_flow_and_escapes_task() -> None:
    controls = ReplayControls(task="Coffee <script>", episode=0)
    controls.set_total_steps(100)
    controls.update_observation(
        episode_step=24,
        cycle=0,
        progress=0.25,
        reward=1.0,
        success=False,
        action_norm=0.75,
    )

    html = _graph_html(controls.snapshot(), "Running")

    assert "DemoActionSource" in html
    assert "RoboCasaAction" in html
    assert "RoboCasaSimulator" in html
    assert "RoboCasaObservation" in html
    assert "GoalSource" in html
    assert "EmbodiedPlanner" in html
    assert "SkillDispatcher" in html
    assert "TaskVerifier" in html
    assert "EventSink" in html
    assert "action 25 of 100" in html
    assert "25.0% complete" in html
    assert "Coffee &lt;script&gt;" in html
    assert "Coffee <script>" not in html


def test_plan_and_events_follow_replay_progress() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    goal = EmbodiedGoal(task="PrepareCoffee", text="Make coffee")
    plan = OfflineEmbodiedPlanner().plan(goal)
    controls.configure_execution(plan.goal, plan)
    controls.update_observation(
        episode_step=400,
        cycle=0,
        progress=0.6,
        reward=0.0,
        success=False,
        action_norm=1.0,
    )

    snapshot = controls.snapshot()
    statuses = [(event.step_id, event.status) for event in snapshot.events]

    assert snapshot.current_step_id == "step-3"
    assert statuses[:2] == [("", "completed"), ("step-1", "running")]
    assert ("step-1", "completed") in statuses
    assert ("step-2", "completed") in statuses
    assert ("step-3", "running") in statuses
    assert "Place the mug under the dispenser" in _plan_html(snapshot)
    assert "Execution events" in _events_html(snapshot)

    controls.update_observation(
        episode_step=748,
        cycle=0,
        progress=1.0,
        reward=1.0,
        success=True,
        action_norm=0.5,
    )
    verified = controls.snapshot()
    assert verified.current_step_id == ""
    assert verified.events[-1].status == "verified"


def test_plan_advances_across_hierarchical_stages() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(task="PrepareCoffee", text="Make coffee")
    )
    controls.configure_execution(plan.goal, plan)

    stages = []
    for step in plan.steps:
        if not stages or stages[-1][0] != step.stage_id:
            stages.append((step.stage_id, []))
        stages[-1][1].append(step)

    seen_stages = []
    completed_step_ids: set[str] = set()
    for stage_id, stage_steps in stages:
        progress = (stage_steps[0].start_fraction + stage_steps[-1].end_fraction) / 2.0
        controls.update_observation(
            episode_step=round(progress * 749),
            cycle=0,
            progress=progress,
            reward=0.0,
            success=False,
            action_norm=1.0,
        )
        snapshot = controls.snapshot()
        active_step = next(
            step for step in plan.steps if step.step_id == snapshot.current_step_id
        )
        seen_stages.append(active_step.stage_id)
        completed_events = {
            event.step_id
            for event in snapshot.events
            if event.status == "completed" and event.step_id
        }
        assert completed_step_ids <= completed_events
        assert any(
            event.step_id == active_step.step_id and event.status == "running"
            for event in snapshot.events
        )
        completed_step_ids.update(step.step_id for step in stage_steps)

    assert seen_stages == [stage_id for stage_id, _ in stages]


def test_success_status_remains_terminal_after_replay_completion() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(task="PrepareCoffee", text="Make coffee")
    )
    controls.configure_execution(plan.goal, plan)
    controls.update_observation(
        episode_step=748,
        cycle=0,
        progress=1.0,
        reward=1.0,
        success=True,
        action_norm=0.5,
    )

    controls.mark_complete()
    controls.set_paused(True)
    controls.set_paused(False)
    controls.request_step()
    controls.update_observation(
        episode_step=749,
        cycle=0,
        progress=1.0,
        reward=0.0,
        success=False,
        action_norm=0.0,
    )

    snapshot = controls.snapshot()
    assert snapshot.status == "Success"
    assert snapshot.success is True
    assert snapshot.paused is False
    assert snapshot.current_step_id == ""
    assert sum(event.status == "verified" for event in snapshot.events) == 1
    assert not any(event.status == "failed" for event in snapshot.events)
    assert controls.claim_next_action() == (True, False)


def test_late_success_replaces_a_stale_terminal_failure() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(task="PrepareCoffee", text="Make coffee")
    )
    controls.configure_execution(plan.goal, plan)
    controls.set_total_steps(750)
    controls.mark_complete()
    controls.update_observation(
        episode_step=749,
        cycle=0,
        progress=1.0,
        reward=0.0,
        success=False,
        action_norm=0.0,
    )

    controls.update_observation(
        episode_step=731,
        cycle=0,
        progress=0.977,
        reward=1.0,
        success=True,
        action_norm=0.5,
    )

    snapshot = controls.snapshot()
    assert snapshot.status == "Success"
    assert snapshot.success is True
    assert not any(event.status == "failed" for event in snapshot.events)
    assert sum(event.status == "verified" for event in snapshot.events) == 1


def test_failed_status_records_unverified_terminal_outcome() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(task="PrepareCoffee", text="Make coffee")
    )
    controls.configure_execution(plan.goal, plan)
    controls.set_total_steps(750)
    controls.mark_complete()

    controls.update_observation(
        episode_step=749,
        cycle=0,
        progress=1.0,
        reward=0.0,
        success=False,
        action_norm=0.0,
    )

    snapshot = controls.snapshot()
    assert snapshot.status == "Failed"
    assert snapshot.current_step_id == ""
    assert snapshot.events[-1].kind == "verification"
    assert snapshot.events[-1].status == "failed"


def test_verification_step_remains_active_until_success() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(task="PrepareCoffee", text="Make coffee")
    )
    controls.configure_execution(plan.goal, plan)
    controls.set_total_steps(749)
    controls.update_observation(
        episode_step=748,
        cycle=0,
        progress=1.0,
        reward=0.0,
        success=False,
        action_norm=0.5,
    )

    snapshot = controls.snapshot()
    verification_step = plan.steps[-1]

    assert verification_step.skill == "verify"
    assert snapshot.current_step_id == verification_step.step_id
    assert not any(
        event.step_id == verification_step.step_id and event.status == "completed"
        for event in snapshot.events
    )

    html = _plan_html(snapshot)
    assert html.count(">Completed<") == len(plan.steps) - 1
    assert ">Running<" in html


def test_latest_goal_wins_when_planners_complete_out_of_order() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    planner = OfflineEmbodiedPlanner()
    older_started = Event()
    release_older = Event()
    older_errors: list[Exception] = []

    def delayed_plan(goal: EmbodiedGoal):
        if goal.text == "older goal":
            older_started.set()
            assert release_older.wait(timeout=2.0)
        return planner.plan(goal)

    def submit_older() -> None:
        try:
            controls.submit_goal("older goal")
        except RuntimeError as exc:
            older_errors.append(exc)

    controls.set_goal_handler(delayed_plan)
    older_thread = Thread(target=submit_older)
    older_thread.start()
    assert older_started.wait(timeout=1.0)

    newest_plan = controls.submit_goal("newest goal")
    release_older.set()
    older_thread.join(timeout=2.0)

    assert newest_plan.goal.text == "newest goal"
    assert controls.snapshot().goal_text == "newest goal"
    assert len(older_errors) == 1
    assert "superseded" in str(older_errors[0]).lower()


def test_bridge_streams_native_robosuite_state(monkeypatch) -> None:
    servers = []
    scenes = []

    class FakeServer:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.stopped = False
            self.connect_callbacks = []
            servers.append(self)

        def on_client_connect(self, callback):
            self.connect_callbacks.append(callback)
            return callback

        def stop(self) -> None:
            self.stopped = True

    class FakeScene:
        def __init__(self, server, model, *, num_envs) -> None:
            self.server = server
            self.model = model
            self.num_envs = num_envs
            self.geom_groups_visible = [True] * 6
            self.synced = False
            self.gui_created = False
            self.gui_kwargs = {}
            self._tracked_body_id = 1
            self.updates = []
            scenes.append(self)

        def _sync_visibilities(self) -> None:
            self.synced = True

        def create_visualization_gui(self, **kwargs) -> None:
            self.gui_created = True
            self.gui_kwargs = kwargs

        def update_from_mjdata(self, data) -> None:
            self.updates.append(data)

    monkeypatch.setitem(sys.modules, "viser", SimpleNamespace(ViserServer=FakeServer))
    monkeypatch.setitem(
        sys.modules,
        "mjviser",
        SimpleNamespace(ViserMujocoScene=FakeScene),
    )

    body_names = ["world", "left_eef_target", "mobilebase0_base", "robot0_link0"]
    model = SimpleNamespace(
        nbody=len(body_names),
        body=lambda body_id: SimpleNamespace(name=body_names[body_id]),
    )
    data = object()
    sim = SimpleNamespace(
        model=SimpleNamespace(_model=model),
        data=SimpleNamespace(_data=data),
    )
    bridge = MjviserBridge(port=0, label="test")

    bridge.start(sim)
    bridge.update(sim)

    assert servers[0].kwargs == {"host": "127.0.0.1", "port": 0, "label": "test"}
    assert scenes[0].model is model
    assert scenes[0].num_envs == 1
    assert scenes[0].geom_groups_visible[0] is False
    assert scenes[0].synced is True
    assert scenes[0].gui_created is True
    assert scenes[0].gui_kwargs == {"camera_distance": 3.0}
    assert scenes[0]._tracked_body_id == 2
    assert scenes[0].updates == [data, data]
    assert len(servers[0].connect_callbacks) == 1

    bridge.close()
    assert servers[0].stopped is True


def test_robot_tracking_body_prefers_mobile_base() -> None:
    names = [
        "world",
        "left_eef_target",
        "robot0_base",
        "robot0_link0",
        "mobilebase0_base",
    ]
    model = SimpleNamespace(
        nbody=len(names),
        body=lambda body_id: SimpleNamespace(name=names[body_id]),
    )

    assert _robot_tracking_body_id(model) == 4
    assert _robot_tracking_body_id(SimpleNamespace(nbody=0)) is None


def test_camera_preset_updates_client_atomically() -> None:
    camera = SimpleNamespace()
    client = SimpleNamespace(camera=camera, atomic=nullcontext)

    _apply_camera_preset(client, _DEFAULT_CAMERA_PRESETS["Robot"])

    assert camera.position == (0.0, -1.0, 2.6)
    assert camera.look_at == (0.0, 0.3, 1.0)
    assert camera.up_direction == (0.0, 0.0, 1.0)
    assert camera.min_orbit_distance == 0.05
    assert camera.max_orbit_distance == 20.0


def test_camera_presets_follow_robot_base_orientation() -> None:
    data = SimpleNamespace(xmat=[np.eye(3)])

    presets = _camera_presets_from_robot(data, 0)

    assert tuple(round(value, 3) for value in presets["Robot"].position) == (
        -1.0,
        0.0,
        2.6,
    )
    assert presets["Agent"].look_at == (1.0, 0.0, 0.8)
    assert presets["Overview"].position == (-0.6, 0.0, 5.0)
