from __future__ import annotations

import sys
from contextlib import nullcontext
from threading import Event, Thread
from time import sleep
from types import SimpleNamespace

import numpy as np
import pytest

from examples.advanced.robocasa.embodied import (
    EmbodiedGoal,
    OfflineEmbodiedPlanner,
    SkillPlan,
    SkillStep,
)
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
    _viewer_url,
)


def test_replay_controls_pause_step_restart_and_speed() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=2)
    controls.set_total_steps(749)

    assert controls.claim_next_action() == (True, None)

    controls.set_paused(True)
    assert controls.claim_next_action() == (False, None)

    controls.request_step()
    assert controls.claim_next_action() == (True, None)
    assert controls.claim_next_action() == (False, None)

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
    assert controls.claim_next_action() == (True, 1)
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

    assert controls.claim_next_action() == (True, 1)
    controls.update_observation(
        episode_step=0,
        cycle=1,
        progress=0.0,
        reward=0.0,
        success=False,
        action_norm=0.25,
    )
    assert controls.snapshot().status == "Running"


def test_restart_ignores_observations_from_future_cycles() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    controls.set_total_steps(10)

    controls.update_observation(
        episode_step=9,
        cycle=1,
        progress=1.0,
        reward=1.0,
        success=True,
        action_norm=0.5,
    )

    snapshot = controls.snapshot()
    assert snapshot.cycle == 0
    assert snapshot.episode_step == 0
    assert snapshot.progress == 0.0
    assert snapshot.success is False


def test_repeated_restart_requests_advance_one_cycle_each() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)

    controls.request_restart()
    controls.request_restart()
    assert controls.snapshot().cycle == 1
    assert controls.claim_next_action() == (True, 1)

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
    assert controls.claim_next_action() == (True, 2)


def test_reinitialize_clears_terminal_telemetry_and_plan_lifecycle() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(task="PrepareCoffee", text="Make coffee")
    )
    controls.configure_execution(plan.goal, plan)
    controls.set_total_steps(10)
    controls.update_observation(
        episode_step=9,
        cycle=0,
        progress=1.0,
        reward=1.0,
        success=True,
        action_norm=0.75,
    )
    controls.mark_complete()

    controls.set_total_steps(20)

    snapshot = controls.snapshot()
    assert snapshot.status == "Ready"
    assert snapshot.paused is False
    assert snapshot.episode_step == 0
    assert snapshot.total_steps == 20
    assert snapshot.progress == 0.0
    assert snapshot.reward == 0.0
    assert snapshot.success is False
    assert snapshot.action_norm == 0.0
    assert snapshot.current_step_id == plan.steps[0].step_id
    assert [(event.step_id, event.status) for event in snapshot.events] == [
        ("", "completed"),
        (plan.steps[0].step_id, "running"),
    ]
    assert controls.claim_next_action() == (True, None)


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
    controls.set_total_steps(749)
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
    controls.set_total_steps(749)

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
    controls.set_total_steps(749)
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
    assert controls.claim_next_action() == (False, None)

    controls.request_restart()
    assert controls.claim_next_action() == (True, 1)


def test_terminal_failure_stops_source_ticks() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    controls.set_total_steps(1)
    controls.update_observation(
        episode_step=0,
        cycle=0,
        progress=1.0,
        reward=0.0,
        success=False,
        action_norm=0.0,
    )
    controls.mark_complete()

    assert controls.snapshot().status == "Failed"
    assert controls.claim_next_action() == (False, None)


def test_late_intermediate_success_does_not_replace_terminal_failure() -> None:
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
    assert snapshot.status == "Failed"
    assert snapshot.success is False
    assert any(event.status == "failed" for event in snapshot.events)
    assert not any(event.status == "verified" for event in snapshot.events)


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
    assert any(
        event.step_id == plan.verification_step_id and event.status == "failed"
        for event in snapshot.events
    )
    assert not any(
        event.step_id != plan.verification_step_id and event.status == "failed"
        for event in snapshot.events
        if event.step_id
    )
    html = _plan_html(snapshot)
    assert ">Failed<" in html
    assert "#b91c1c" in html


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
    assert html.count(">Passed<") >= len(plan.steps) - 1
    assert ">Current<" in html


def test_terminal_failure_without_verify_step_is_run_level_only() -> None:
    goal = EmbodiedGoal(task="PrepareCoffee", text="Place the mug")
    plan = SkillPlan(
        goal=goal,
        steps=(
            SkillStep(
                step_id="place-mug",
                skill="place",
                label="Place the mug",
            ),
        ),
    )
    controls = ReplayControls(task=goal.task, episode=goal.episode)
    controls.configure_execution(goal, plan)
    controls.set_total_steps(1)
    controls.update_observation(
        episode_step=0,
        cycle=0,
        progress=1.0,
        reward=0.0,
        success=False,
        action_norm=0.0,
    )
    controls.mark_complete()

    snapshot = controls.snapshot()
    assert snapshot.status == "Failed"
    assert snapshot.current_step_id == ""
    assert not any(
        event.step_id == "place-mug" and event.status == "failed"
        for event in snapshot.events
    )
    assert any(
        event.kind == "verification"
        and event.step_id == ""
        and event.status == "failed"
        for event in snapshot.events
    )


def test_step_lifecycle_events_are_not_lost_from_the_current_ui_contract() -> None:
    goal = EmbodiedGoal(task="PrepareCoffee", text="Run a long plan")
    count = 70
    plan = SkillPlan(
        goal=goal,
        steps=tuple(
            SkillStep(
                step_id=f"step-{index}",
                skill="locate",
                label=f"Locate item {index}",
                start_fraction=index / count,
                end_fraction=(index + 1) / count,
                depends_on=(f"step-{index - 1}",) if index else (),
            )
            for index in range(count)
        ),
    )
    controls = ReplayControls(task=goal.task, episode=goal.episode)
    controls.configure_execution(goal, plan)
    controls.set_total_steps(count)

    for index in range(1, count):
        controls.update_observation(
            episode_step=index,
            cycle=0,
            progress=index / count,
            reward=0.0,
            success=False,
            action_norm=0.0,
        )

    snapshot = controls.snapshot()
    assert len(snapshot.events) > 64
    assert any(
        event.step_id == "step-0" and event.status == "completed"
        for event in snapshot.events
    )
    assert "Locate item 0" in _plan_html(snapshot)


def test_replay_controls_allow_only_one_active_planner_request() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    planner = OfflineEmbodiedPlanner()
    older_started = Event()
    release_older = Event()
    older_plans: list[SkillPlan] = []

    def delayed_plan(goal: EmbodiedGoal):
        if goal.text == "older goal":
            older_started.set()
            assert release_older.wait(timeout=2.0)
        return planner.plan(goal)

    def submit_older() -> None:
        older_plans.append(controls.submit_goal("older goal"))

    controls.set_goal_handler(delayed_plan)
    older_thread = Thread(target=submit_older)
    older_thread.start()
    assert older_started.wait(timeout=1.0)

    with pytest.raises(RuntimeError, match="already running"):
        controls.submit_goal("newest goal")
    release_older.set()
    older_thread.join(timeout=2.0)

    assert len(older_plans) == 1
    assert controls.snapshot().goal_text == "older goal"
    newest_plan = controls.submit_goal("newest goal")
    assert newest_plan.goal.text == "newest goal"
    assert controls.snapshot().goal_text == "newest goal"
    assert controls.snapshot().planning is False


def test_timed_out_planner_cannot_restart_the_replay_later() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    planner = OfflineEmbodiedPlanner()
    started = Event()
    release = Event()

    def delayed_plan(goal: EmbodiedGoal):
        if goal.text == "Make coffee":
            started.set()
            assert release.wait(timeout=2.0)
        return planner.plan(goal)

    controls.set_goal_handler(delayed_plan)
    with pytest.raises(RuntimeError, match="timed out"):
        controls.submit_goal("Make coffee", timeout=0.01)
    assert started.wait(timeout=1.0)
    assert controls.snapshot().planning is True

    with pytest.raises(RuntimeError, match="already running"):
        controls.submit_goal("Prepare tea")
    release.set()
    for _ in range(100):
        if not controls.snapshot().planning:
            break
        sleep(0.01)

    assert controls.snapshot().planning is False
    assert controls.snapshot().goal_text != "Make coffee"
    assert controls.claim_next_action()[0] is True


def test_cancelled_planner_result_is_not_published() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    planner = OfflineEmbodiedPlanner()
    started = Event()
    release = Event()
    errors: list[Exception] = []

    def delayed_plan(goal: EmbodiedGoal):
        started.set()
        assert release.wait(timeout=2.0)
        return planner.plan(goal)

    def submit() -> None:
        try:
            controls.submit_goal("Make coffee")
        except RuntimeError as exc:
            errors.append(exc)

    controls.set_goal_handler(delayed_plan)
    thread = Thread(target=submit)
    thread.start()
    assert started.wait(timeout=1.0)
    controls.cancel_pending_goals()
    assert controls.snapshot().planning is False
    release.set()
    thread.join(timeout=2.0)

    assert len(errors) == 1
    assert "cancelled" in str(errors[0]).lower()
    assert controls.snapshot().goal_text == ""


def test_terminal_failure_cannot_be_rewritten_by_late_success() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    controls.set_total_steps(1)
    controls.update_observation(
        episode_step=0,
        cycle=0,
        progress=1.0,
        reward=0.0,
        success=False,
        action_norm=0.0,
    )
    controls.mark_complete()
    controls.update_observation(
        episode_step=1,
        cycle=0,
        progress=1.0,
        reward=1.0,
        success=True,
        action_norm=0.5,
    )

    snapshot = controls.snapshot()
    assert snapshot.status == "Failed"
    assert snapshot.success is False


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
            self.site_groups_visible = [True] * 6
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
    assert scenes[0].geom_groups_visible[1] is True
    assert scenes[0].geom_groups_visible[2] is False
    assert scenes[0].site_groups_visible[0] is False
    assert scenes[0].site_groups_visible[1] is True
    assert scenes[0].site_groups_visible[2] is False
    assert scenes[0].synced is True
    assert scenes[0].gui_created is True
    assert scenes[0].gui_kwargs == {"camera_distance": 3.0}
    assert scenes[0]._tracked_body_id == 2
    assert scenes[0].updates == [data, data]
    assert len(servers[0].connect_callbacks) == 1

    bridge.close()
    assert servers[0].stopped is True


def test_bridge_cleans_up_after_partial_start_failure(monkeypatch) -> None:
    servers = []

    class FakeServer:
        def __init__(self, **kwargs) -> None:
            self.stopped = False
            servers.append(self)

        def stop(self) -> None:
            self.stopped = True

    class FailingScene:
        def __init__(self, server, model, *, num_envs) -> None:
            raise RuntimeError("scene setup failed")

    monkeypatch.setitem(sys.modules, "viser", SimpleNamespace(ViserServer=FakeServer))
    monkeypatch.setitem(
        sys.modules,
        "mjviser",
        SimpleNamespace(ViserMujocoScene=FailingScene),
    )
    sim = SimpleNamespace(
        model=SimpleNamespace(_model=object()),
        data=SimpleNamespace(_data=object()),
    )
    bridge = MjviserBridge(port=0, label="test")

    with pytest.raises(RuntimeError, match="scene setup failed"):
        bridge.start(sim)

    assert servers[0].stopped is True
    assert bridge._server is None
    assert bridge._scene is None
    bridge.close()


def test_bridge_close_is_idempotent_when_server_stop_fails() -> None:
    class FailingServer:
        def __init__(self) -> None:
            self.stop_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1
            raise RuntimeError("stop failed")

    server = FailingServer()
    bridge = MjviserBridge(port=0, label="test")
    bridge._server = server
    bridge._scene = object()

    with pytest.raises(RuntimeError, match="stop failed"):
        bridge.close()

    assert server.stop_calls == 1
    assert bridge._server is None
    assert bridge._scene is None
    bridge.close()


def test_camera_preset_survives_disconnected_and_reconnected_clients() -> None:
    class FakeServer:
        def __init__(self) -> None:
            self.clients = {}
            self.connect_callbacks = []
            self.stopped = False

        def get_clients(self):
            return self.clients

        def on_client_connect(self, callback):
            self.connect_callbacks.append(callback)
            return callback

        def stop(self) -> None:
            self.stopped = True

    def client():
        return SimpleNamespace(camera=SimpleNamespace(), atomic=nullcontext)

    server = FakeServer()
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    bridge = MjviserBridge(port=0, label="test", controls=controls)
    bridge._server = server
    bridge._scene = object()
    bridge._register_camera_handler()

    bridge.apply_camera_preset("Overview")
    assert controls.snapshot().camera_preset == "Overview"
    first_client = client()
    server.connect_callbacks[0](first_client)
    assert first_client.camera.position == _DEFAULT_CAMERA_PRESETS["Overview"].position

    server.clients.clear()
    bridge.apply_camera_preset("Robot")
    assert controls.snapshot().camera_preset == "Robot"
    second_client = client()
    server.connect_callbacks[0](second_client)
    assert second_client.camera.position == _DEFAULT_CAMERA_PRESETS["Robot"].position

    bridge.close()
    assert server.stopped is True


def test_refresh_controls_keeps_viser_panel_in_sync() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    controls.set_total_steps(10)
    controls.set_paused(True)
    controls.set_speed(0.5)
    controls.set_camera_preset("Overview")

    bridge = MjviserBridge(port=0, label="test", controls=controls)
    bridge._status_markdown = SimpleNamespace(content="")
    bridge._plan_html = SimpleNamespace(content="")
    bridge._graph_html = SimpleNamespace(content="")
    bridge._events_html = SimpleNamespace(content="")
    bridge._progress = SimpleNamespace(value=0.0)
    bridge._goal_input = SimpleNamespace(value="")
    bridge._planner_dropdown = SimpleNamespace(value="gemini")
    bridge._execution_mode_dropdown = SimpleNamespace(value="live_planning")
    bridge._camera_group = SimpleNamespace(value="Robot")
    bridge._speed_dropdown = SimpleNamespace(value="1x")
    bridge._run_goal_button = SimpleNamespace(disabled=False)
    bridge._pause_button = SimpleNamespace(disabled=False)
    bridge._resume_button = SimpleNamespace(disabled=True)
    bridge._step_button = SimpleNamespace(disabled=True)
    bridge._restart_button = SimpleNamespace(disabled=True)

    bridge.refresh_controls()

    assert "**Paused**" in bridge._status_markdown.content
    assert bridge._planner_dropdown.value == "offline"
    assert bridge._execution_mode_dropdown.value == "demonstration"
    assert bridge._camera_group.value == "Overview"
    assert bridge._speed_dropdown.value == "0.5x"
    assert bridge._pause_button.disabled is True
    assert bridge._resume_button.disabled is False
    assert bridge._step_button.disabled is False
    assert bridge._restart_button.disabled is False

    controls.update_observation(
        episode_step=9,
        cycle=0,
        progress=1.0,
        reward=1.0,
        success=True,
        action_norm=0.25,
    )
    bridge.refresh_controls()

    assert "**Success**" in bridge._status_markdown.content
    assert bridge._pause_button.disabled is True
    assert bridge._resume_button.disabled is True
    assert bridge._step_button.disabled is True


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
    preset = _DEFAULT_CAMERA_PRESETS["Robot"]

    _apply_camera_preset(client, preset)

    assert camera.position == preset.position
    assert camera.look_at == preset.look_at
    assert camera.up_direction == (0.0, 0.0, 1.0)
    assert camera.min_orbit_distance == 0.05
    assert camera.max_orbit_distance == 20.0


def test_camera_presets_follow_robot_base_orientation() -> None:
    data = SimpleNamespace(
        xmat=[np.eye(3)],
        xpos=[np.array([10.0, 20.0, 30.0])],
    )

    presets = _camera_presets_from_robot(data, 0)

    assert tuple(round(value, 3) for value in presets["Robot"].position) == (
        9.0,
        20.0,
        32.6,
    )
    assert presets["Agent"].look_at == (11.0, 20.0, 30.8)
    assert presets["Overview"].position == (9.4, 20.0, 35.0)


@pytest.mark.parametrize(
    ("host", "expected"),
    [
        ("127.0.0.1", "http://localhost:8080"),
        ("0.0.0.0", "http://localhost:8080"),
        ("::", "http://localhost:8080"),
        ("::1", "http://localhost:8080"),
        ("2001:db8::1", "http://[2001:db8::1]:8080"),
        ("simulator.local", "http://simulator.local:8080"),
    ],
)
def test_viewer_url_is_browser_safe(host: str, expected: str) -> None:
    assert _viewer_url(host, 8080) == expected
