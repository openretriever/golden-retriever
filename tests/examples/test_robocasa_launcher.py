from __future__ import annotations

import importlib.util
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


def test_discovery_always_includes_curated_tasks(monkeypatch, tmp_path: Path) -> None:
    installed = tmp_path / "TurnOnMicrowave"
    installed.mkdir()
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
    assert all(scene.unavailable_reason for scene in scenes[:-1])


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

    stopped = manager.stop()
    assert processes[1].terminated is True
    assert stopped["state"] == "idle"


def test_viewer_manager_checks_selected_interface_port(monkeypatch) -> None:
    checked_ports = []

    class RunningProcess:
        returncode = None

        def poll(self):
            return None

    monkeypatch.setattr(
        launcher,
        "_port_is_open",
        lambda _host, port: checked_ports.append(port) or True,
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

    assert checked_ports == [8085, 8086]


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
