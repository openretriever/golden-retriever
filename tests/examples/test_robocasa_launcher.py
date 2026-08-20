from __future__ import annotations

from types import SimpleNamespace

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


def test_viewer_command_runs_existing_retriever_robocasa_app() -> None:
    manager = ViewerManager(host="127.0.0.1", port=8085, duration=180)

    command = manager.command(task=SCENE.task, split=SCENE.split, episode=3)

    assert command[1:3] == ["-m", "examples.advanced.robocasa.app"]
    assert command[command.index("--task") + 1] == "TurnOnMicrowave"
    assert command[command.index("--episode") + 1] == "3"
    assert command[command.index("--visualize") + 1] == "mjviser"
    assert command[command.index("--viser-port") + 1] == "8085"


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


def test_launcher_lists_installed_scenes(monkeypatch) -> None:
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
            stop=lambda: None,
        ),
    )
    app = launcher.create_app(viewer_host="127.0.0.1", viewer_port=8085, duration=180)

    with TestClient(app) as client:
        response = client.get("/api/scenes")
        index = client.get("/")
        missing = client.post("/api/launch", json={"task": "Missing", "episode": 0})

    assert response.status_code == 200
    assert response.json()[0]["task"] == "TurnOnMicrowave"
    assert "Installed scenes" in index.text
    assert missing.status_code == 404
