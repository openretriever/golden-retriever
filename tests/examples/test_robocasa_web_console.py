from __future__ import annotations

import json
from threading import Event
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from examples.advanced.robocasa import web_console
from examples.advanced.robocasa.embodied import EmbodiedGoal, OfflineEmbodiedPlanner
from examples.advanced.robocasa.mjviser_bridge import ReplayControls


def _planned_controls() -> ReplayControls:
    controls = ReplayControls(task="PrepareCoffee", episode=3)
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(text="Make coffee", task="PrepareCoffee", episode=3)
    )
    controls.configure_execution(plan.goal, plan)
    return controls


def test_snapshot_payload_serializes_nested_plan_events_and_control_state() -> None:
    controls = _planned_controls()
    controls.set_total_steps(200)
    controls.set_speed(1.5)
    controls.set_paused(True)
    controls.update_observation(
        episode_step=100,
        cycle=2,
        progress=0.5,
        reward=0.25,
        success=False,
        action_norm=1.75,
    )

    payload = web_console.snapshot_payload(controls.snapshot())

    encoded = json.loads(json.dumps(payload))
    assert payload["task"] == "PrepareCoffee"
    assert payload["episode"] == 3
    assert payload["paused"] is True
    assert payload["speed"] == 1.5
    assert payload["episode_step"] == 100
    assert payload["total_steps"] == 200
    assert payload["plan"]["source"] == "offline"
    assert payload["plan"]["goal"] == {
        "text": "Make coffee",
        "task": "PrepareCoffee",
        "episode": 3,
        "planner": "offline",
        "execution_mode": "demonstration",
    }
    assert encoded["plan"]["steps"][1]["depends_on"] == ["step-1"]
    assert payload["events"][0]["kind"] == "dispatch"
    assert payload["events"][1]["step_id"] == "step-1"


def test_handle_command_updates_replay_controls() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(controls, "http://viewer.test")

    paused = console.handle_command("/api/pause", {})
    assert paused["ok"] is True
    assert paused["state"]["paused"] is True
    assert controls.claim_next_action() == (False, False)

    resumed = console.handle_command("/api/resume", {})
    assert resumed["state"]["paused"] is False
    assert controls.claim_next_action() == (True, False)

    console.handle_command("/api/pause", {})
    stepped = console.handle_command("/api/step", {})
    assert stepped["state"]["paused"] is True
    assert stepped["state"]["status"] == "Stepping"
    assert controls.claim_next_action() == (True, False)
    assert controls.claim_next_action() == (False, False)

    controls.update_observation(
        episode_step=12,
        cycle=1,
        progress=0.4,
        reward=0.5,
        success=False,
        action_norm=2.0,
    )
    restarted = console.handle_command("/api/restart", {})
    assert restarted["state"]["status"] == "Restarting"
    assert restarted["state"]["progress"] == 0.0
    assert controls.claim_next_action() == (True, True)

    sped_up = console.handle_command("/api/speed", {"speed": 2.0})
    assert sped_up["state"]["speed"] == 2.0


@pytest.mark.parametrize("speed", [0, -0.25, "fast", None])
def test_handle_command_rejects_invalid_speed(speed: object) -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(controls, "http://viewer.test")

    with pytest.raises(ValueError):
        console.handle_command("/api/speed", {"speed": speed})


def test_goal_camera_and_invalid_commands() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=4)
    controls.set_goal_handler(OfflineEmbodiedPlanner().plan)
    selected_cameras: list[str] = []
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        camera_handler=selected_cameras.append,
    )

    goal_result = console.handle_command(
        "/api/goal",
        {"text": "Make coffee", "planner": "offline"},
    )
    assert goal_result["state"]["goal_text"] == "Make coffee"
    assert goal_result["state"]["planner"] == "offline"
    assert goal_result["state"]["plan"]["source"] == "offline"
    assert "4 subplans and 5 allow-listed skills" in goal_result["reply"]
    assert "pinned demonstrated plan" in goal_result["reply"]
    assert "Restarting episode 4" in goal_result["reply"]
    assert goal_result["plan_summary"] == {
        "task": "PrepareCoffee",
        "source": "offline",
        "steps": 5,
        "stages": 4,
        "execution_mode": "demonstration",
    }
    assert controls.claim_next_action() == (True, True)

    camera_result = console.handle_command(
        "/api/camera",
        {"preset": "Overview"},
    )
    assert selected_cameras == ["Overview"]
    assert camera_result["ok"] is True

    with pytest.raises(ValueError):
        console.handle_command("/api/goal", {"text": "   "})
    with pytest.raises(ValueError):
        console.handle_command("/api/camera", {"preset": "Ceiling"})
    with pytest.raises(ValueError):
        console.handle_command("/api/camera", {})
    with pytest.raises(ValueError):
        console.handle_command("/api/unknown", {})


def test_live_planning_goal_keeps_verified_trajectory_semantics() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=1)
    controls.set_goal_handler(OfflineEmbodiedPlanner().plan)
    console = web_console.RetrieverWebConsole(controls, "http://viewer.test")

    result = console.handle_command(
        "/api/goal",
        {
            "text": "Prepare coffee in several steps",
            "planner": "offline",
            "execution_mode": "live_planning",
        },
    )

    assert result["state"]["execution_mode"] == "live_planning"
    assert result["state"]["plan"]["goal"]["execution_mode"] == "live_planning"
    assert result["plan_summary"]["execution_mode"] == "live_planning"
    assert "skill plan was generated now" in result["reply"]
    assert "verified trajectory" in result["reply"]


def test_goal_command_rejects_unknown_execution_mode() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=1)
    controls.set_goal_handler(OfflineEmbodiedPlanner().plan)
    console = web_console.RetrieverWebConsole(controls, "http://viewer.test")

    with pytest.raises(ValueError, match="execution mode"):
        console.handle_command(
            "/api/goal",
            {
                "text": "Make coffee",
                "execution_mode": "unrestricted_actions",
            },
        )


def test_lifecycle_uses_server_without_binding_a_port(monkeypatch) -> None:
    servers: list[FakeServer] = []

    class FakeServer:
        def __init__(self, address: tuple[str, int], handler: type[Any]) -> None:
            self.requested_address = address
            self.handler = handler
            self.server_address = (address[0], 43123)
            self.started = Event()
            self.shutdown_called = False
            self.close_called = False
            servers.append(self)

        def serve_forever(self, **_kwargs: Any) -> None:
            self.started.set()

        def shutdown(self) -> None:
            self.shutdown_called = True

        def server_close(self) -> None:
            self.close_called = True

    monkeypatch.setattr(web_console, "ThreadingHTTPServer", FakeServer)
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
    )

    console.start()

    assert servers[0].started.wait(timeout=1.0)
    assert servers[0].requested_address == ("127.0.0.1", 0)
    assert console.url == "http://localhost:43123"

    console.close()

    assert servers[0].shutdown_called is True
    assert servers[0].close_called is True


def test_http_rejects_non_object_json_with_structured_error() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
    )
    console.start()
    request = Request(
        f"{console.url}/api/goal",
        data=b"[]",
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2.0)
        assert error.value.code == 400
        payload = json.loads(error.value.read())
        assert payload == {"error": "Request body must be a JSON object"}
    finally:
        console.close()
