"""Local scene launcher for Retriever-connected RoboCasa replays."""

from __future__ import annotations

import argparse
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from threading import RLock
from typing import Any, Literal

try:
    from pydantic import BaseModel, Field
except ImportError:
    BaseModel = None
    Field = None


@dataclass(frozen=True)
class Scene:
    task: str
    kind: str
    split: str
    episodes: int
    horizon: int
    preview: str
    available: bool = True
    unavailable_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


if BaseModel is not None and Field is not None:

    class LaunchRequest(BaseModel):
        task: str
        episode: int = Field(default=0, ge=0)
        goal: str = Field(default="", max_length=2_000)
        planner: Literal["offline", "gemini"] = "offline"
        execution_mode: Literal["demonstration", "live_planning"] = "demonstration"
        interface: Literal["console", "viser"] = "console"

else:
    LaunchRequest = None


_PREVIEWS = {
    "TurnOnMicrowave": "/media/turn-on-microwave-replay.jpg",
    "PrepareCoffee": "/media/retriever-robocasa-web-console.jpg",
}

_CURATED_TASKS = (
    ("PrepareCoffee", "Composite"),
    ("DeliverStraw", "Composite"),
    ("OpenDrawer", "Atomic"),
    ("OpenCabinet", "Atomic"),
    ("PackIdenticalLunches", "Composite"),
    ("LoadDishwasher", "Composite"),
    ("LoadFridgeByType", "Composite"),
    ("ArrangeDrinkware", "Composite"),
    ("OrganizeCondiments", "Composite"),
    ("StackBowlsCabinet", "Composite"),
    ("RestockPantry", "Composite"),
    ("MicrowaveCorrectMeal", "Composite"),
    ("PrepareToast", "Composite"),
    ("SetupFrying", "Composite"),
    ("CoffeeSetupMug", "Atomic"),
    ("StartCoffeeMachine", "Atomic"),
    ("TurnOnMicrowave", "Atomic"),
)
_CURATED_ORDER = {task: index for index, (task, _kind) in enumerate(_CURATED_TASKS)}
_DEFAULT_PREVIEW = "/media/turn-on-microwave-replay.jpg"
_MISSING_DATASET = "Human demonstration dataset is not installed."


def discover_scenes() -> list[Scene]:
    """Return installed datasets plus the always-visible curated task catalog."""

    try:
        from robocasa.utils.dataset_registry import (
            ATOMIC_TASK_DATASETS,
            COMPOSITE_TASK_DATASETS,
        )
        from robocasa.utils.dataset_registry_utils import get_ds_meta
    except ImportError:
        return [
            _unavailable_scene(task, kind, "RoboCasa is not installed.")
            for task, kind in _CURATED_TASKS
        ]

    scenes_by_task: dict[str, Scene] = {}
    groups = (
        ("Atomic", ATOMIC_TASK_DATASETS),
        ("Composite", COMPOSITE_TASK_DATASETS),
    )
    for kind, registry in groups:
        for task in registry:
            try:
                metadata = get_ds_meta(task=task, split="pretrain", source="human")
                path_value = metadata.get("path") if metadata else None
                dataset_path = Path(path_value) if path_value else None
            except (KeyError, OSError, TypeError, ValueError):
                metadata = None
                dataset_path = None
            available = dataset_path is not None and dataset_path.is_dir()
            if not available and task not in _CURATED_ORDER:
                continue
            scenes_by_task[task] = Scene(
                task=task,
                kind=kind,
                split="pretrain",
                episodes=100,
                horizon=int(metadata.get("horizon", 0)) if metadata else 0,
                preview=_PREVIEWS.get(task, _DEFAULT_PREVIEW),
                available=available,
                unavailable_reason=None if available else _MISSING_DATASET,
            )

    for task, kind in _CURATED_TASKS:
        scenes_by_task.setdefault(
            task,
            _unavailable_scene(task, kind, _MISSING_DATASET),
        )

    return sorted(
        scenes_by_task.values(),
        key=lambda scene: (
            _CURATED_ORDER.get(scene.task, len(_CURATED_ORDER)),
            scene.kind != "Atomic",
            scene.task,
        ),
    )


def _unavailable_scene(task: str, kind: str, reason: str) -> Scene:
    return Scene(
        task=task,
        kind=kind,
        split="pretrain",
        episodes=100,
        horizon=0,
        preview=_PREVIEWS.get(task, _DEFAULT_PREVIEW),
        available=False,
        unavailable_reason=reason,
    )


class ViewerManager:
    """Own one task-specific Retriever console and simulator process."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        duration: float,
        console_port: int | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.console_port = int(console_port if console_port is not None else port + 1)
        self.duration = float(duration)
        self._lock = RLock()
        self._process: subprocess.Popen[Any] | None = None
        self._task: str | None = None
        self._episode = 0
        self._goal = ""
        self._planner = "offline"
        self._execution_mode = "demonstration"
        self._interface: Literal["console", "viser"] = "console"

    @property
    def viewer_url(self) -> str:
        display_host = (
            "localhost" if self.host in {"0.0.0.0", "127.0.0.1"} else self.host
        )
        return f"http://{display_host}:{self.port}"

    @property
    def console_url(self) -> str:
        display_host = (
            "localhost" if self.host in {"0.0.0.0", "127.0.0.1"} else self.host
        )
        return f"http://{display_host}:{self.console_port}"

    def command(
        self,
        *,
        task: str,
        split: str,
        episode: int,
        goal: str = "",
        planner: Literal["offline", "gemini"] = "offline",
        execution_mode: Literal["demonstration", "live_planning"] = "demonstration",
        interface: Literal["console", "viser"] = "console",
    ) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "examples.advanced.robocasa.app",
            "--mode",
            "robocasa",
            "--task",
            task,
            "--split",
            split,
            "--episode",
            str(episode),
            "--goal",
            goal.strip() or task,
            "--planner",
            planner,
            "--execution-mode",
            execution_mode,
            "--seconds",
            str(self.duration),
            "--hz",
            "20",
            "--visualize",
            "mjviser",
            "--viser-host",
            self.host,
            "--viser-port",
            str(self.port),
            "--console-host",
            self.host,
            "--console-port",
            str(self.console_port),
            "--print-every",
            "100",
        ]
        if interface == "viser":
            command.append("--native-viser-controls")
        return command

    def launch(
        self,
        scene: Scene,
        episode: int,
        *,
        goal: str = "",
        planner: Literal["offline", "gemini"] = "offline",
        execution_mode: Literal["demonstration", "live_planning"] = "demonstration",
        interface: Literal["console", "viser"] = "console",
    ) -> dict[str, Any]:
        if not scene.available:
            raise ValueError(scene.unavailable_reason or "Dataset is unavailable.")
        if not 0 <= episode < scene.episodes:
            raise ValueError(f"Episode must be between 0 and {scene.episodes - 1}")
        with self._lock:
            self._stop_locked()
            self._process = subprocess.Popen(
                self.command(
                    task=scene.task,
                    split=scene.split,
                    episode=episode,
                    goal=goal,
                    planner=planner,
                    execution_mode=execution_mode,
                    interface=interface,
                ),
                cwd=Path(__file__).resolve().parents[3],
            )
            self._task = scene.task
            self._episode = episode
            self._goal = goal.strip() or scene.task
            self._planner = planner
            self._execution_mode = execution_mode
            self._interface = interface
            return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            self._stop_locked()
            return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            if process is None:
                state = "idle"
            elif process.poll() is not None:
                state = "stopped" if process.returncode == 0 else "failed"
            elif _port_is_open(
                self.host,
                self.port if self._interface == "viser" else self.console_port,
            ):
                state = "ready"
            else:
                state = "starting"
            return {
                "state": state,
                "task": self._task,
                "episode": self._episode,
                "goal": self._goal,
                "planner": self._planner,
                "execution_mode": self._execution_mode,
                "interface": self._interface,
                "viewer_url": self.viewer_url,
                "console_url": self.console_url,
                "open_url": (
                    self.viewer_url if self._interface == "viser" else self.console_url
                ),
            }

    def _stop_locked(self) -> None:
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        self._process = None
        self._task = None
        self._episode = 0
        self._goal = ""
        self._planner = "offline"
        self._execution_mode = "demonstration"
        self._interface = "console"


def _port_is_open(host: str, port: int) -> bool:
    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    connection = HTTPConnection(connect_host, port, timeout=0.2)
    try:
        connection.request("GET", "/")
        return connection.getresponse().status < 500
    except (HTTPException, OSError):
        return False
    finally:
        connection.close()


def create_app(
    *,
    viewer_host: str,
    viewer_port: int,
    duration: float,
    console_port: int | None = None,
):
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse
    from fastapi.staticfiles import StaticFiles

    if LaunchRequest is None:
        raise RuntimeError("The RoboCasa launcher requires the web extra.")

    scenes = discover_scenes()
    by_task = {scene.task: scene for scene in scenes}
    manager = ViewerManager(
        host=viewer_host,
        port=viewer_port,
        duration=duration,
        console_port=console_port,
    )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        manager.stop()

    app = FastAPI(title="Retriever RoboCasa Scenes", lifespan=lifespan)
    static_dir = Path(__file__).with_name("static")
    media_dir = Path(__file__).resolve().parents[3] / "docs-site/public/media/robocasa"
    app.mount("/media", StaticFiles(directory=media_dir), name="media")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return (static_dir / "launcher.html").read_text(encoding="utf-8")

    @app.get("/api/scenes")
    async def list_scenes() -> list[dict[str, Any]]:
        return [scene.to_dict() for scene in scenes]

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return manager.status()

    @app.post("/api/launch")
    async def launch(request: LaunchRequest) -> dict[str, Any]:
        scene = by_task.get(request.task)
        if scene is None:
            raise HTTPException(status_code=404, detail="Unknown RoboCasa task")
        if not scene.available:
            raise HTTPException(
                status_code=409,
                detail=scene.unavailable_reason or "Dataset is unavailable.",
            )
        try:
            return manager.launch(
                scene,
                request.episode,
                goal=request.goal,
                planner=request.planner,
                execution_mode=request.execution_mode,
                interface=request.interface,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/api/stop")
    async def stop() -> dict[str, Any]:
        return manager.stop()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse installed RoboCasa scenes.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8084)
    parser.add_argument("--viewer-host", default="127.0.0.1")
    parser.add_argument("--viewer-port", type=int, default=8085)
    parser.add_argument("--console-port", type=int, default=8086)
    parser.add_argument("--duration", type=float, default=180.0)
    return parser.parse_args()


def main() -> None:
    import uvicorn

    args = parse_args()
    app = create_app(
        viewer_host=args.viewer_host,
        viewer_port=args.viewer_port,
        duration=args.duration,
        console_port=args.console_port,
    )
    print(f"Retriever RoboCasa scenes: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
