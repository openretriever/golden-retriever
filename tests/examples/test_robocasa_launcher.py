from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

FASTAPI_AVAILABLE = importlib.util.find_spec("fastapi") is not None
if FASTAPI_AVAILABLE:
    from fastapi.testclient import TestClient

from examples.advanced.robocasa import launcher
from examples.advanced.robocasa.launcher import Scene, ViewerManager

SCENE = Scene(
    task="TurnOnMicrowave",
    kind="Atomic",
    split="pretrain",
    episodes=100,
    horizon=500,
    preview="/media/turn-on-microwave-replay.jpg",
)


def test_launcher_template_clamps_episode_and_announces_status() -> None:
    template = (
        Path(launcher.__file__).parent / "static" / "launcher.html"
    ).read_text(encoding="utf-8")

    assert 'role="status" aria-live="polite"' in template
    assert "const maxEpisode = Math.max(0, scene.episodes - 1);" in template
    assert "Math.min(Math.max(0, Number(episode.value) || 0), maxEpisode)" in template


def test_discovery_always_includes_curated_tasks(monkeypatch, tmp_path: Path) -> None:
    installed = tmp_path / "TurnOnMicrowave"
    installed.mkdir()
    (installed / "info.json").write_text(
        '{"total_episodes": 12}',
        encoding="utf-8",
    )
    atomic = {
        "CoffeeSetupMug": {},
        "StartCoffeeMachine": {},
        "TurnOnMicrowave": {},
        "NotInstalled": {},
    }
    composite = {"PrepareCoffee": {}}

    def get_ds_meta(*, task, split, source):
        assert split == "pretrain"
        assert source == "human"
        if task == "TurnOnMicrowave":
            return {"path": str(installed), "horizon": 500}
        return None

    monkeypatch.setitem(
        sys.modules,
        "robocasa.utils.dataset_registry",
        SimpleNamespace(
            ATOMIC_TASK_DATASETS=atomic,
            COMPOSITE_TASK_DATASETS=composite,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "robocasa.utils.dataset_registry_utils",
        SimpleNamespace(get_ds_meta=get_ds_meta),
    )

    scenes = launcher.discover_scenes()

    assert [scene.task for scene in scenes] == [
        "PrepareCoffee",
        "DeliverStraw",
        "OpenDrawer",
        "OpenCabinet",
        "PackIdenticalLunches",
        "LoadDishwasher",
        "LoadFridgeByType",
        "ArrangeDrinkware",
        "OrganizeCondiments",
        "StackBowlsCabinet",
        "RestockPantry",
        "MicrowaveCorrectMeal",
        "PrepareToast",
        "SetupFrying",
        "CoffeeSetupMug",
        "StartCoffeeMachine",
        "TurnOnMicrowave",
    ]
    assert scenes[-1].available is True
    assert scenes[-1].episodes == 12
    assert all(scene.unavailable_reason for scene in scenes[:-1])


def test_dataset_episode_count_prefers_registry_metadata(tmp_path: Path) -> None:
    (tmp_path / "info.json").write_text(
        '{"total_episodes": 12}',
        encoding="utf-8",
    )

    assert launcher._dataset_episode_count({"num_demos": 7}, tmp_path) == 7
    assert launcher._dataset_episode_count({}, tmp_path) == 12
    assert launcher._dataset_episode_count({}, tmp_path / "missing") == 0


def test_viewer_command_runs_existing_retriever_robocasa_app() -> None:
    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)

    command = manager.command(
        task=SCENE.task,
        split=SCENE.split,
        episode=3,
        goal="Warm the soup",
        planner="gemini",
        interface="viser",
    )

    assert command[1:3] == ["-m", "examples.advanced.robocasa.app"]
    assert command[command.index("--task") + 1] == "TurnOnMicrowave"
    assert command[command.index("--episode") + 1] == "3"
    assert command[command.index("--visualize") + 1] == "mjviser"
    assert command[command.index("--viser-port") + 1] == "8085"
    assert command[command.index("--goal") + 1] == "Warm the soup"
    assert command[command.index("--planner") + 1] == "gemini"
    assert "--native-viser-controls" in command
    assert command[0] == sys.executable


def test_viewer_manager_replaces_and_stops_child(monkeypatch) -> None:
    processes = []

    class FakeProcess:
        def __init__(self, command, **kwargs) -> None:
            self.command = command
            self.kwargs = kwargs
            self.returncode = None
            self.terminated = False
            processes.append(self)

        def poll(self):
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            self.returncode = 0

        def wait(self, timeout):
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(launcher.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(launcher, "_port_is_open", lambda _host, _port: False)
    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)

    assert manager.launch(SCENE, 0)["state"] == "starting"
    assert manager.launch(SCENE, 1)["episode"] == 1
    assert processes[0].terminated is True
    assert processes[0].kwargs["start_new_session"] is (os.name == "posix")

    stopped = manager.stop()
    assert processes[1].terminated is True
    assert stopped["state"] == "idle"


def test_viewer_manager_retains_child_when_it_cannot_be_reaped() -> None:
    class StuckProcess:
        returncode = None

        def poll(self):
            return None

        def terminate(self) -> None:
            return None

        def wait(self, timeout):
            raise launcher.subprocess.TimeoutExpired("robocasa", timeout)

        def kill(self) -> None:
            return None

    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)
    manager._process = StuckProcess()
    manager._task = "PrepareCoffee"

    status = manager.stop()

    assert manager._process is not None
    assert status["state"] == "failed"
    assert status["task"] == "PrepareCoffee"
    assert status["error"]


def test_viewer_manager_checks_selected_interface_port(monkeypatch) -> None:
    checked_ports = []
    matched_runs = []

    class RunningProcess:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(
        launcher,
        "_port_is_open",
        lambda _host, port: checked_ports.append(port) or True,
    )
    monkeypatch.setattr(
        launcher,
        "_console_matches_run",
        lambda host, port, *, task, episode, launch_id: (
            matched_runs.append((host, port, task, episode, launch_id)) or True
        ),
    )
    manager = ViewerManager(
        host="127.0.0.1",
        port=8085,
        console_port=8086,
        duration=180,
    )
    manager._process = RunningProcess()

    manager._interface = "viser"
    assert manager.status()["state"] == "ready"
    manager._interface = "console"
    assert manager.status()["state"] == "ready"

    assert checked_ports == [8085]
    assert matched_runs == [
        ("127.0.0.1", 8086, None, 0, ""),
        ("127.0.0.1", 8086, None, 0, ""),
    ]


def test_viewer_manager_cleans_surviving_process_group(monkeypatch) -> None:
    signals = []
    waited_for = []

    class ExitedParent:
        pid = 4321
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(launcher.os, "name", "posix")
    monkeypatch.setattr(
        launcher,
        "_process_group_exists",
        lambda process_group_id: process_group_id == 4321,
    )
    monkeypatch.setattr(
        launcher,
        "_signal_process",
        lambda process, sig, *, process_group_id=None: signals.append(
            (process, sig, process_group_id)
        ),
    )
    monkeypatch.setattr(
        launcher,
        "_wait_for_process_group_exit",
        lambda process_group_id: waited_for.append(process_group_id) or True,
    )
    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)
    parent = ExitedParent()
    manager._process = parent
    manager._process_group_id = parent.pid

    manager.stop()

    assert signals == [
        (parent, launcher.signal.SIGTERM, 4321),
        (parent, launcher.signal.SIGKILL, 4321),
    ]
    assert waited_for == [4321]


def test_viewer_manager_retains_process_group_when_cleanup_times_out(
    monkeypatch,
) -> None:
    class ExitedParent:
        pid = 4321
        returncode = 0

        def poll(self):
            return 0

    monkeypatch.setattr(launcher.os, "name", "posix")
    monkeypatch.setattr(launcher, "_process_group_exists", lambda _pgid: True)
    monkeypatch.setattr(launcher, "_signal_process", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        launcher,
        "_wait_for_process_group_exit",
        lambda _process_group_id: False,
    )
    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)
    parent = ExitedParent()
    manager._process = parent
    manager._process_group_id = parent.pid

    status = manager.stop()

    assert status["state"] == "failed"
    assert status["error"] == "Simulator process group could not be stopped"
    assert manager._process is parent


def test_viewer_manager_waits_for_matching_console(monkeypatch) -> None:
    checked_runs = []

    class RunningProcess:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(
        launcher,
        "_console_matches_run",
        lambda host, port, *, task, episode, launch_id: (
            checked_runs.append((host, port, task, episode, launch_id)) or False
        ),
    )
    manager = ViewerManager(
        host="127.0.0.1",
        port=8085,
        console_port=8086,
        duration=180,
    )
    manager._process = RunningProcess()
    manager._task = "PrepareCoffee"
    manager._episode = 3
    manager._launch_id = "run-123"
    manager._interface = "console"

    assert manager.status()["state"] == "starting"
    assert checked_runs == [("127.0.0.1", 8086, "PrepareCoffee", 3, "run-123")]


def test_viewer_manager_reports_child_exit_code() -> None:
    class FailedProcess:
        returncode = 7

        def poll(self):
            return self.returncode

    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)
    manager._process = FailedProcess()
    manager._task = "PrepareCoffee"

    status = manager.status()

    assert status["state"] == "failed"
    assert status["task"] == "PrepareCoffee"
    assert status["error"] == "Simulator process exited with code 7."


def test_viewer_manager_treats_clean_child_exit_as_stopped() -> None:
    class CompletedProcess:
        returncode = 0

        def poll(self):
            return self.returncode

    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)
    manager._process = CompletedProcess()

    status = manager.status()

    assert status["state"] == "stopped"
    assert status["error"] is None


def test_viewer_command_keeps_standalone_console_renderer_neutral() -> None:
    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)

    command = manager.command(
        task=SCENE.task,
        split=SCENE.split,
        episode=0,
        interface="console",
    )

    assert "--native-viser-controls" not in command


def test_viewer_command_forwards_live_planning_mode() -> None:
    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)

    command = manager.command(
        task=SCENE.task,
        split=SCENE.split,
        episode=0,
        execution_mode="live_planning",
        interface="console",
    )

    assert command[command.index("--execution-mode") + 1] == "live_planning"


def test_viewer_command_forwards_launch_identity() -> None:
    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)

    command = manager.command(
        task=SCENE.task,
        split=SCENE.split,
        episode=0,
        launch_id="run-123",
    )

    assert command[command.index("--launch-id") + 1] == "run-123"


def test_viewer_manager_rejects_unavailable_scene(monkeypatch) -> None:
    monkeypatch.setattr(
        launcher.subprocess,
        "Popen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("spawned")),
    )
    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)
    unavailable = Scene(
        task="PrepareCoffee",
        kind="Composite",
        split="pretrain",
        episodes=100,
        horizon=0,
        preview="/media/retriever-robocasa-web-console.jpg",
        available=False,
        unavailable_reason="Dataset missing",
    )

    try:
        manager.launch(unavailable, 0)
    except ValueError as exc:
        assert str(exc) == "Dataset missing"
    else:
        raise AssertionError("Unavailable scene was launched")


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI extra is not installed")
def test_launcher_lists_installed_scenes(monkeypatch) -> None:
    stops = []
    monkeypatch.setattr(launcher, "discover_scenes", lambda: [SCENE])
    monkeypatch.setattr(
        launcher,
        "ViewerManager",
        lambda **_kwargs: SimpleNamespace(
            status=lambda: {
                "state": "idle",
                "task": None,
                "episode": 0,
                "viewer_url": "http://localhost:8085",
            },
            stop=lambda: stops.append(True),
            launch=lambda *_args, **_kwargs: {},
        ),
    )
    app = launcher.create_app(viewer_host="127.0.0.1", viewer_port=8085, duration=180)

    with TestClient(app) as client:
        response = client.get("/api/scenes")
        index = client.get("/")
        missing = client.post("/api/launch", json={"task": "Missing", "episode": 0})

    assert response.status_code == 200
    assert response.json()[0]["task"] == "TurnOnMicrowave"
    assert "Task catalog" in index.text
    assert 'id="goal"' in index.text
    assert 'id="planner"' in index.text
    assert "function applyStatus" in index.text
    assert "expectedGeneration === requestGeneration" in index.text
    assert missing.status_code == 404
    assert stops == [True]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI extra is not installed")
def test_launcher_reports_unavailable_dataset_without_launching(monkeypatch) -> None:
    unavailable = Scene(
        task="PrepareCoffee",
        kind="Composite",
        split="pretrain",
        episodes=100,
        horizon=0,
        preview="/media/retriever-robocasa-web-console.jpg",
        available=False,
        unavailable_reason="Install the human demonstration dataset.",
    )
    monkeypatch.setattr(launcher, "discover_scenes", lambda: [unavailable])
    monkeypatch.setattr(
        launcher,
        "ViewerManager",
        lambda **_kwargs: SimpleNamespace(
            status=dict,
            stop=lambda: None,
            launch=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("launch called")
            ),
        ),
    )
    app = launcher.create_app(viewer_host="127.0.0.1", viewer_port=8085, duration=180)

    with TestClient(app) as client:
        scenes = client.get("/api/scenes")
        response = client.post(
            "/api/launch",
            json={"task": "PrepareCoffee", "goal": "Make coffee", "planner": "offline"},
        )

    assert scenes.json()[0]["available"] is False
    assert scenes.json()[0]["unavailable_reason"]
    assert response.status_code == 409


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI extra is not installed")
def test_launcher_forwards_goal_and_planner(monkeypatch) -> None:
    launches = []
    manager = SimpleNamespace(
        status=dict,
        stop=lambda: None,
        launch=lambda scene, episode, **kwargs: (
            launches.append((scene.task, episode, kwargs)) or {"state": "starting"}
        ),
    )
    monkeypatch.setattr(launcher, "discover_scenes", lambda: [SCENE])
    monkeypatch.setattr(launcher, "ViewerManager", lambda **_kwargs: manager)
    app = launcher.create_app(viewer_host="127.0.0.1", viewer_port=8085, duration=180)

    with TestClient(app) as client:
        response = client.post(
            "/api/launch",
            json={
                "task": "TurnOnMicrowave",
                "episode": 2,
                "goal": "Heat dinner",
                "planner": "gemini",
                "execution_mode": "live_planning",
                "interface": "viser",
            },
        )

    assert response.status_code == 200
    assert launches == [
        (
            "TurnOnMicrowave",
            2,
            {
                "goal": "Heat dinner",
                "planner": "gemini",
                "execution_mode": "live_planning",
                "interface": "viser",
            },
        )
    ]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI extra is not installed")
def test_launcher_allows_same_origin_and_originless_posts(monkeypatch) -> None:
    stops = []
    monkeypatch.setattr(launcher, "discover_scenes", lambda: [SCENE])
    monkeypatch.setattr(
        launcher,
        "ViewerManager",
        lambda **_kwargs: SimpleNamespace(
            status=dict,
            stop=lambda: stops.append(True) or {"state": "idle"},
            launch=lambda *_args, **_kwargs: {"state": "starting"},
        ),
    )
    app = launcher.create_app(viewer_host="127.0.0.1", viewer_port=8085, duration=180)

    with TestClient(app, base_url="http://localhost:8084") as client:
        same_origin = client.post(
            "/api/stop",
            headers={"Origin": "http://localhost:8084"},
        )
        originless = client.post("/api/stop")

    assert same_origin.status_code == 200
    assert originless.status_code == 200
    assert stops == [True, True, True]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI extra is not installed")
@pytest.mark.parametrize(
    "origin",
    ["https://example.com", "null", "http://localhost:not-a-port"],
)
def test_launcher_rejects_cross_origin_posts(monkeypatch, origin: str) -> None:
    stops = []
    monkeypatch.setattr(launcher, "discover_scenes", lambda: [SCENE])
    monkeypatch.setattr(
        launcher,
        "ViewerManager",
        lambda **_kwargs: SimpleNamespace(
            status=dict,
            stop=lambda: stops.append(True) or {"state": "idle"},
            launch=lambda *_args, **_kwargs: {"state": "starting"},
        ),
    )
    app = launcher.create_app(viewer_host="127.0.0.1", viewer_port=8085, duration=180)

    with TestClient(app, base_url="http://localhost:8084") as client:
        response = client.post(
            "/api/stop",
            headers={"Origin": origin},
        )

    assert response.status_code == 403
    assert (
        response.json()["detail"] == "Cross-origin launcher requests are not allowed."
    )
    assert stops == [True]


@pytest.mark.skipif(not FASTAPI_AVAILABLE, reason="FastAPI extra is not installed")
def test_launcher_requires_json_for_post_bodies(monkeypatch) -> None:
    stops = []
    monkeypatch.setattr(launcher, "discover_scenes", lambda: [SCENE])
    monkeypatch.setattr(
        launcher,
        "ViewerManager",
        lambda **_kwargs: SimpleNamespace(
            status=dict,
            stop=lambda: stops.append(True) or {"state": "idle"},
            launch=lambda *_args, **_kwargs: {"state": "starting"},
        ),
    )
    app = launcher.create_app(viewer_host="127.0.0.1", viewer_port=8085, duration=180)

    with TestClient(app, base_url="http://localhost:8084") as client:
        response = client.post(
            "/api/stop",
            content="stop",
            headers={"Content-Type": "text/plain"},
        )

    assert response.status_code == 415
    assert response.json()["detail"] == "POST request bodies must use application/json."
    assert stops == [True]
