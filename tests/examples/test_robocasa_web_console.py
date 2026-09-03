from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from threading import Event
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from examples.advanced.robocasa import web_console
from examples.advanced.robocasa.embodied import EmbodiedGoal, OfflineEmbodiedPlanner
from examples.advanced.robocasa.runtime import ReplayControls

CONSOLE_HTML = (
    Path(__file__).parents[2]
    / "examples"
    / "advanced"
    / "robocasa"
    / "static"
    / "console.html"
)


def _planned_controls() -> ReplayControls:
    controls = ReplayControls(task="PrepareCoffee", episode=3)
    plan = OfflineEmbodiedPlanner().plan(
        EmbodiedGoal(text="Make coffee", task="PrepareCoffee", episode=3)
    )
    controls.configure_execution(plan.goal, plan)
    return controls


def test_console_plan_uses_lifecycle_events_as_authoritative_state() -> None:
    html = CONSOLE_HTML.read_text()

    assert "function renderPlan(plan, progress, currentStepId, events)" in html
    assert 'event.status === "completed" && event.step_id' in html
    assert 'event.status === "failed" && event.step_id' in html
    assert (
        "renderPlan(state.plan, progress, state.current_step_id, state.events)" in html
    )


def test_snapshot_payload_serializes_nested_plan_events_and_control_state() -> None:
    controls = _planned_controls()
    controls.set_total_steps(200)
    controls.set_speed(1.5)
    controls.set_paused(True)
    controls.update_observation(
        episode_step=100,
        cycle=0,
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
    assert payload["status"] == "Paused"
    assert payload["presentation"] == {
        "status": "Paused",
        "tone": "paused",
        "terminal": False,
    }
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


def test_snapshot_payload_uses_canonical_planning_presentation() -> None:
    controls = _planned_controls()

    payload = web_console.snapshot_payload(replace(controls.snapshot(), planning=True))

    assert payload["status"] == "Planning"
    assert payload["presentation"]["tone"] == "running"
    assert payload["terminal"] is False


def test_goal_capability_reflects_connected_planner() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)

    assert controls.can_submit_goals is False

    controls.set_goal_handler(OfflineEmbodiedPlanner().plan)

    assert controls.can_submit_goals is True


def test_handle_command_updates_replay_controls() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(controls, "http://viewer.test")

    paused = console.handle_command("/api/pause", {})
    assert paused["ok"] is True
    assert paused["state"]["paused"] is True
    assert controls.claim_next_action() == (False, None)

    resumed = console.handle_command("/api/resume", {})
    assert resumed["state"]["paused"] is False
    assert controls.claim_next_action() == (True, None)

    console.handle_command("/api/pause", {})
    stepped = console.handle_command("/api/step", {})
    assert stepped["state"]["paused"] is True
    assert stepped["state"]["status"] == "Stepping"
    assert controls.claim_next_action() == (True, None)
    assert controls.claim_next_action() == (False, None)

    controls.update_observation(
        episode_step=12,
        cycle=0,
        progress=0.4,
        reward=0.5,
        success=False,
        action_norm=2.0,
    )
    assert controls.snapshot().progress == 0.4
    restarted = console.handle_command("/api/restart", {})
    assert restarted["state"]["status"] == "Restarting"
    assert restarted["state"]["progress"] == 0.0
    assert controls.claim_next_action() == (True, 1)

    sped_up = console.handle_command("/api/speed", {"speed": 2.0})
    assert sped_up["state"]["speed"] == 2.0


def test_camera_command_updates_shared_runtime_state() -> None:
    selected: list[str] = []
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        camera_handler=selected.append,
    )

    result = console.handle_command("/api/camera", {"preset": "Overview"})

    assert selected == ["Overview"]
    assert result["state"]["camera_preset"] == "Overview"
    assert controls.snapshot().camera_preset == "Overview"


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
    assert "recorded RoboCasa data" in goal_result["reply"]
    assert "Restarting episode 4" in goal_result["reply"]
    assert goal_result["plan_summary"] == {
        "task": "PrepareCoffee",
        "source": "offline",
        "steps": 5,
        "stages": 4,
        "execution_mode": "demonstration",
    }
    assert controls.claim_next_action() == (True, 1)

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


def test_dynamic_plan_goal_keeps_recorded_replay_semantics() -> None:
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
    assert "recorded replay data" in result["reply"]


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


def test_goal_command_rejects_oversized_text() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=1)
    controls.set_goal_handler(OfflineEmbodiedPlanner().plan)
    console = web_console.RetrieverWebConsole(controls, "http://viewer.test")

    with pytest.raises(ValueError, match="at most 2000"):
        console.handle_command("/api/goal", {"text": "x" * 2_001})


def test_goal_command_classifies_planner_availability_without_string_matching() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=1)
    console = web_console.RetrieverWebConsole(controls, "http://viewer.test")

    with pytest.raises(web_console._ServiceUnavailableError, match="planner"):
        console.handle_command("/api/goal", {"text": "Make coffee"})


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


def test_viewer_readiness_requires_a_healthy_http_response(monkeypatch) -> None:
    class Response:
        status = 200

        def read(self, _size: int) -> bytes:
            return b"x"

    class Connection:
        def __init__(self, host: str, port: int, timeout: float) -> None:
            assert (host, port, timeout) == ("viewer.test", 8085, 0.25)

        def request(self, method: str, path: str) -> None:
            assert (method, path) == ("GET", "/scene")

        def getresponse(self) -> Response:
            return Response()

        def close(self) -> None:
            return None

    monkeypatch.setattr(web_console, "HTTPConnection", Connection)
    controls = ReplayControls(task="PrepareCoffee", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test:8085/scene",
    )

    assert console.viewer_is_ready() is True


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
        headers={
            "Content-Type": "application/json",
            "X-Retriever-Token": console.control_token,
            "X-Retriever-Launch-ID": console.launch_id,
        },
        method="POST",
    )
    try:
        with pytest.raises(HTTPError) as error:
            urlopen(request, timeout=2.0)
        assert error.value.code == 400
        payload = json.loads(error.value.read())
        assert payload == {
            "error": "Request body must be a JSON object",
            "code": "invalid_request",
        }
    finally:
        console.close()


def test_http_page_binds_commands_to_this_console_run() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
        launch_id="run-123",
    )
    console.start()
    try:
        html = urlopen(console.url, timeout=2.0).read().decode()
        assert "__RETRIEVER_CONTROL_TOKEN__" not in html
        assert "__RETRIEVER_LAUNCH_ID__" not in html
        assert console.control_token in html
        assert 'const pageLaunchId = "run-123"' in html
    finally:
        console.close()


def test_http_rejects_missing_token_and_stale_launch() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
        launch_id="run-current",
    )
    console.start()
    try:
        missing = Request(
            f"{console.url}/api/pause",
            data=b"{}",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as error:
            urlopen(missing, timeout=2.0)
        assert error.value.code == 403

        status, payload = _post_error(
            console,
            "/api/pause",
            headers={
                "Content-Type": "application/json",
                "X-Retriever-Launch-ID": "run-old",
            },
        )
        assert (status, payload["code"]) == (409, "stale_launch")
    finally:
        console.close()


def test_http_rejects_oversized_request_body() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
    )
    console.start()
    try:
        status, payload = _post_error(
            console,
            "/api/goal",
            data=b"x" * (64 * 1024 + 1),
        )
        assert (status, payload["code"]) == (413, "request_too_large")
    finally:
        console.close()


def _post_error(
    console: web_console.RetrieverWebConsole,
    path: str,
    *,
    data: bytes = b"{}",
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, Any]]:
    request_headers = {
        "X-Retriever-Token": console.control_token,
        "X-Retriever-Launch-ID": console.launch_id,
    }
    if headers is None:
        request_headers["Content-Type"] = "application/json"
    else:
        request_headers.update(headers)
    request = Request(
        f"{console.url}{path}",
        data=data,
        headers=request_headers,
        method="POST",
    )
    with pytest.raises(HTTPError) as error:
        urlopen(request, timeout=2.0)
    return error.value.code, json.loads(error.value.read())


def test_http_post_requires_same_origin_when_origin_is_present() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
    )
    console.start()
    try:
        status, payload = _post_error(
            console,
            "/api/pause",
            headers={
                "Content-Type": "application/json",
                "Origin": "https://attacker.example",
            },
        )
        assert status == 403
        assert payload["code"] == "cross_origin_forbidden"

        request = Request(
            f"{console.url}/api/pause",
            data=b"{}",
            headers={
                "Content-Type": "application/json; charset=utf-8",
                "Origin": console.url,
                "X-Retriever-Token": console.control_token,
                "X-Retriever-Launch-ID": console.launch_id,
            },
            method="POST",
        )
        response = json.loads(urlopen(request, timeout=2.0).read())
        assert response["state"]["paused"] is True
    finally:
        console.close()


def test_http_post_allows_programmatic_client_without_origin() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
    )
    console.start()
    try:
        request = Request(
            f"{console.url}/api/pause",
            data=b"{}",
            headers={
                "Content-Type": "application/json",
                "X-Retriever-Token": console.control_token,
                "X-Retriever-Launch-ID": console.launch_id,
            },
            method="POST",
        )
        response = json.loads(urlopen(request, timeout=2.0).read())
        assert response["state"]["paused"] is True
    finally:
        console.close()


@pytest.mark.parametrize("content_type", [None, "text/plain", "application/xml"])
def test_http_post_requires_json_content_type(content_type: str | None) -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
    )
    console.start()
    try:
        headers = {"Content-Type": content_type} if content_type else {}
        status, payload = _post_error(
            console,
            "/api/pause",
            headers=headers,
        )
        assert status == 415
        assert payload["code"] == "unsupported_media_type"
    finally:
        console.close()


def test_http_distinguishes_validation_unknown_and_unavailable_errors() -> None:
    controls = ReplayControls(task="TurnOnMicrowave", episode=0)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
    )
    console.start()
    try:
        status, payload = _post_error(
            console,
            "/api/speed",
            data=b'{"speed": "fast"}',
        )
        assert (status, payload["code"]) == (400, "invalid_request")

        status, payload = _post_error(
            console,
            "/api/pause",
            data=b"{not-json}",
        )
        assert (status, payload["code"]) == (400, "invalid_request")

        status, payload = _post_error(console, "/api/unknown")
        assert (status, payload["code"]) == (404, "unknown_command")

        status, payload = _post_error(
            console,
            "/api/camera",
            data=b'{"preset": "Robot"}',
        )
        assert (status, payload["code"]) == (503, "service_unavailable")

        status, payload = _post_error(
            console,
            "/api/goal",
            data=b'{"text": "Make coffee"}',
        )
        assert (status, payload["code"]) == (503, "service_unavailable")
    finally:
        console.close()


def test_http_planner_exception_is_structured_and_server_stays_available() -> None:
    controls = ReplayControls(task="PrepareCoffee", episode=0)

    def broken_planner(_goal: EmbodiedGoal) -> None:
        raise OSError("planner backend disconnected")

    controls.set_goal_handler(broken_planner)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
    )
    console.start()
    try:
        status, payload = _post_error(
            console,
            "/api/goal",
            data=b'{"text": "Make coffee"}',
        )
        assert status == 502
        assert payload == {
            "error": "Planner failed: planner backend disconnected",
            "code": "planner_error",
        }

        state = json.loads(urlopen(f"{console.url}/api/state", timeout=2.0).read())
        assert state["task"] == "PrepareCoffee"
    finally:
        console.close()


def test_http_planner_timeout_returns_gateway_timeout() -> None:
    release_planner = Event()
    controls = ReplayControls(task="PrepareCoffee", episode=0)

    def blocked_planner(_goal: EmbodiedGoal) -> None:
        release_planner.wait(timeout=1.0)

    controls.set_goal_handler(blocked_planner)
    console = web_console.RetrieverWebConsole(
        controls,
        "http://viewer.test",
        host="127.0.0.1",
        port=0,
        planner_timeout=0.01,
    )
    console.start()
    try:
        status, payload = _post_error(
            console,
            "/api/goal",
            data=b'{"text": "Make coffee"}',
        )
        assert (status, payload["code"]) == (504, "planner_timeout")
    finally:
        release_planner.set()
        console.close()
