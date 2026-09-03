"""Local scene launcher for Retriever-connected RoboCasa replays."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import ipaddress
import json
import logging
import os
import signal
import subprocess
import sys
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, replace
from http import HTTPStatus
from http.client import HTTPConnection, HTTPException
from pathlib import Path
from threading import RLock
from time import monotonic, sleep
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from .baselines import BaselineSpec, discover_method_baselines

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
    label: str = ""
    provider: str = "RoboCasa"
    runner: str = "robocasa"
    description: str = ""
    source_url: str = ""
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
_MAX_REQUEST_BYTES = 64 * 1024
_LOGGER = logging.getLogger(__name__)


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
                episodes=_dataset_episode_count(metadata, dataset_path),
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


def discover_baseline_scenes(*, horizon: int = 1200) -> list[Scene]:
    """Project typed baseline manifests into the shared launcher catalog."""

    return [
        _baseline_scene(spec, horizon=horizon) for spec in discover_method_baselines()
    ]


def _baseline_scene(spec: BaselineSpec, *, horizon: int) -> Scene:
    return Scene(
        task=spec.baseline_id,
        kind=spec.tier,
        split=spec.environment,
        episodes=1,
        horizon=horizon,
        preview="",
        label=spec.label,
        provider=spec.family,
        runner=spec.runner,
        description=spec.description,
        source_url=spec.reference_url,
        available=spec.available,
        unavailable_reason=spec.unavailable_reason,
    )


def _unavailable_scene(task: str, kind: str, reason: str) -> Scene:
    return Scene(
        task=task,
        kind=kind,
        split="pretrain",
        episodes=0,
        horizon=0,
        preview=_PREVIEWS.get(task, _DEFAULT_PREVIEW),
        available=False,
        unavailable_reason=reason,
    )


def _dataset_episode_count(
    metadata: dict[str, Any] | None,
    dataset_path: Path | None,
) -> int:
    """Read an installed dataset's episode count without opening its payload."""

    for key in ("total_episodes", "num_episodes", "num_demos", "episodes"):
        value = metadata.get(key) if metadata else None
        if isinstance(value, int) and value > 0:
            return value

    if dataset_path is not None:
        for info_path in (
            dataset_path / "meta" / "info.json",
            dataset_path / "info.json",
        ):
            try:
                info = json.loads(info_path.read_text(encoding="utf-8"))
            except (FileNotFoundError, OSError, TypeError, ValueError):
                continue
            for key in ("total_episodes", "num_episodes", "num_demos", "episodes"):
                value = info.get(key)
                if isinstance(value, int) and value > 0:
                    return value
    return 0


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
        self._process_group_id: int | None = None
        self._task: str | None = None
        self._episode = 0
        self._goal = ""
        self._planner = "offline"
        self._execution_mode = "demonstration"
        self._interface: Literal["console", "viser"] = "console"
        self._runner = "robocasa"
        self._launch_id = ""
        self._last_error: str | None = None

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
        launch_id: str = "",
        runner: str = "robocasa",
    ) -> list[str]:
        if runner == "robosuite_lift":
            steps = max(1, int(self.duration / 0.05))
            return [
                sys.executable,
                "-m",
                "examples.advanced.robocasa.robosuite_lift",
                "--mode",
                "robosuite",
                "--env",
                "Lift",
                "--visualize",
                "mjviser",
                "--viser-host",
                self.host,
                "--viser-port",
                str(self.port),
                "--steps",
                str(steps),
                "--dt",
                "0.05",
                "--print-every",
                "100",
                "--harness",
            ]
        if runner != "robocasa":
            raise ValueError(f"Unknown simulator runner: {runner}")
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
            "--launch-id",
            launch_id,
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
        if scene.episodes > 0 and not 0 <= episode < scene.episodes:
            raise ValueError(f"Episode must be between 0 and {scene.episodes - 1}")
        with self._lock:
            if not self._stop_locked():
                raise RuntimeError(
                    self._last_error or "Could not stop the active simulator"
                )
            self._task = scene.task
            self._episode = episode
            self._goal = goal.strip() or scene.task
            self._planner = planner
            self._execution_mode = execution_mode
            self._interface = "viser" if scene.runner != "robocasa" else interface
            self._runner = scene.runner
            self._launch_id = uuid4().hex
            try:
                self._process = subprocess.Popen(
                    self.command(
                        task=scene.task,
                        split=scene.split,
                        episode=episode,
                        goal=goal,
                        planner=planner,
                        execution_mode=execution_mode,
                        interface=interface,
                        launch_id=self._launch_id,
                        runner=scene.runner,
                    ),
                    cwd=Path(__file__).resolve().parents[3],
                    start_new_session=os.name == "posix",
                )
                if os.name == "posix":
                    self._process_group_id = getattr(self._process, "pid", None)
            except OSError as exc:
                self._process_group_id = None
                self._last_error = str(exc)
                raise RuntimeError(f"Could not start simulator: {exc}") from exc
        return self.status()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._stop_locked():
                raise RuntimeError(
                    self._last_error or "Could not stop the active simulator"
                )
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            process = self._process
            returncode = process.poll() if process is not None else None
            if returncode not in {None, 0} and self._last_error is None:
                self._last_error = f"Simulator process exited with code {returncode}."
            status = {
                "task": self._task,
                "episode": self._episode,
                "goal": self._goal,
                "planner": self._planner,
                "execution_mode": self._execution_mode,
                "interface": self._interface,
                "runner": self._runner,
                "launch_id": self._launch_id,
                "viewer_url": self.viewer_url,
                "console_url": self.console_url,
                "open_url": (
                    self.viewer_url
                    if self._interface == "viser" or self._runner != "robocasa"
                    else self.console_url
                ),
                "error": self._last_error,
            }
        if process is None:
            state = "failed" if status["error"] else "idle"
        elif status["error"]:
            state = "failed"
        elif returncode is not None:
            state = "stopped"
        elif (
            status["runner"] != "robocasa"
            and _port_is_open(self.host, self.port)
            or _console_matches_run(
                self.host,
                self.console_port,
                task=status["task"],
                episode=status["episode"],
                launch_id=status["launch_id"],
            )
            and _port_is_open(self.host, self.port)
        ):
            state = "ready"
        else:
            state = "starting"
        return {"state": state, **status}

    def _stop_locked(self) -> bool:
        process = self._process
        process_group_id = self._process_group_id
        if process is not None and (
            process.poll() is None or process_group_id is not None
        ):
            _signal_process(
                process,
                signal.SIGTERM,
                process_group_id=process_group_id,
            )
        if process is not None and process.poll() is None:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                _signal_process(
                    process,
                    signal.SIGKILL,
                    process_group_id=process_group_id,
                )
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired as exc:
                    self._last_error = f"Simulator process could not be stopped: {exc}"
                    return False
        if process_group_id is not None and _process_group_exists(process_group_id):
            _signal_process(
                process,
                signal.SIGKILL,
                process_group_id=process_group_id,
            )
            if not _wait_for_process_group_exit(process_group_id):
                self._last_error = "Simulator process group could not be stopped"
                return False

        self._process = None
        self._process_group_id = None
        self._task = None
        self._episode = 0
        self._goal = ""
        self._planner = "offline"
        self._execution_mode = "demonstration"
        self._interface = "console"
        self._runner = "robocasa"
        self._launch_id = ""
        self._last_error = None
        return True


def _signal_process(
    process: subprocess.Popen[Any],
    sig: signal.Signals,
    *,
    process_group_id: int | None = None,
) -> None:
    if os.name == "posix" and process_group_id is not None:
        try:
            os.killpg(process_group_id, sig)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    if sig == signal.SIGTERM:
        process.terminate()
    else:
        process.kill()


def _process_group_exists(process_group_id: int) -> bool:
    if os.name != "posix":
        return False
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _wait_for_process_group_exit(
    process_group_id: int,
    *,
    timeout: float = 5.0,
) -> bool:
    deadline = monotonic() + timeout
    while _process_group_exists(process_group_id):
        if monotonic() >= deadline:
            return False
        sleep(0.05)
    return True


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


def _console_matches_run(
    host: str,
    port: int,
    *,
    task: str | None,
    episode: int,
    launch_id: str,
) -> bool:
    """Reject a stale console left over from a previous task or episode."""

    connect_host = "127.0.0.1" if host == "0.0.0.0" else host
    connection = HTTPConnection(connect_host, port, timeout=0.2)
    try:
        connection.request("GET", "/api/config")
        response = connection.getresponse()
        if response.status != HTTPStatus.OK:
            return False
        payload = json.loads(response.read())
        return (
            payload.get("task") == task
            and payload.get("episode") == episode
            and payload.get("launch_id") == launch_id
        )
    except (HTTPException, json.JSONDecodeError, OSError, TypeError, ValueError):
        return False
    finally:
        connection.close()


def _request_origin(request: Any) -> tuple[str, str, int | None] | None:
    parsed = urlsplit(str(request.base_url))
    try:
        return parsed.scheme.lower(), (parsed.hostname or "").lower(), parsed.port
    except ValueError:
        return None


def _header_origin(value: str) -> tuple[str, str, int | None] | None:
    parsed = urlsplit(value)
    if (
        parsed.scheme.lower() not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        return None
    try:
        return parsed.scheme.lower(), parsed.hostname.lower(), parsed.port
    except ValueError:
        return None


def _host_is_allowed(value: str, configured_host: str) -> bool:
    parsed = urlsplit(f"http://{value}")
    hostname = (parsed.hostname or "").lower()
    if hostname in {"localhost", "127.0.0.1", "::1"}:
        return True
    configured = configured_host.lower().strip("[]")
    try:
        address = ipaddress.ip_address(hostname)
        return address.is_loopback or hostname == configured
    except ValueError:
        return configured not in {"", "0.0.0.0", "::"} and hostname == configured


def _post_has_body(request: Any) -> bool:
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            return int(content_length) > 0
        except ValueError:
            return True
    return bool(request.headers.get("transfer-encoding"))


def create_app(
    *,
    viewer_host: str,
    viewer_port: int,
    duration: float,
    console_port: int | None = None,
    launcher_host: str = "127.0.0.1",
):
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles

    if LaunchRequest is None:
        raise RuntimeError("The RoboCasa launcher requires the web extra.")

    baseline_horizon = max(1, int(duration / 0.05))
    scenes = [
        *discover_baseline_scenes(horizon=baseline_horizon),
        *discover_scenes(),
    ]
    manager = ViewerManager(
        host=viewer_host,
        port=viewer_port,
        duration=duration,
        console_port=console_port,
    )

    @asynccontextmanager
    async def lifespan(_app):
        yield
        try:
            await asyncio.to_thread(manager.stop)
        except RuntimeError as exc:
            _LOGGER.warning(
                "Simulator cleanup failed during launcher shutdown: %s", exc
            )

    app = FastAPI(title="Retriever Embodied Scenes", lifespan=lifespan)
    control_token = uuid4().hex
    app.state.control_token = control_token
    static_dir = Path(__file__).with_name("static")
    media_dir = Path(__file__).resolve().parents[3] / "docs-site/public/media/robocasa"
    if media_dir.is_dir():
        app.mount("/media", StaticFiles(directory=media_dir), name="media")
    else:
        scenes = [replace(scene, preview="") for scene in scenes]
    by_task = {scene.task: scene for scene in scenes}

    @app.middleware("http")
    async def protect_local_mutations(request: Request, call_next):
        if request.method == "POST":
            if not _host_is_allowed(request.headers.get("host", ""), launcher_host):
                return JSONResponse(
                    status_code=HTTPStatus.FORBIDDEN,
                    content={
                        "detail": "Launcher commands require a local or explicitly configured host."
                    },
                )
            origin = request.headers.get("origin")
            if origin is not None:
                supplied_origin = _header_origin(origin)
                if supplied_origin is None or supplied_origin != _request_origin(
                    request
                ):
                    return JSONResponse(
                        status_code=HTTPStatus.FORBIDDEN,
                        content={
                            "detail": "Cross-origin launcher requests are not allowed."
                        },
                    )
            supplied_token = request.headers.get("x-retriever-token", "")
            if not hmac.compare_digest(supplied_token, control_token):
                return JSONResponse(
                    status_code=HTTPStatus.FORBIDDEN,
                    content={"detail": "Launcher control token is missing or invalid."},
                )
            content_length = request.headers.get("content-length")
            if content_length is not None:
                try:
                    request_bytes = int(content_length)
                except ValueError:
                    request_bytes = _MAX_REQUEST_BYTES + 1
                if request_bytes < 0 or request_bytes > _MAX_REQUEST_BYTES:
                    return JSONResponse(
                        status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        content={
                            "detail": f"Request body must be at most {_MAX_REQUEST_BYTES} bytes."
                        },
                    )
            body = await request.body()
            if len(body) > _MAX_REQUEST_BYTES:
                return JSONResponse(
                    status_code=HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                    content={
                        "detail": f"Request body must be at most {_MAX_REQUEST_BYTES} bytes."
                    },
                )
            if _post_has_body(request):
                media_type = request.headers.get("content-type", "").split(";", 1)[0]
                if media_type.strip().lower() != "application/json":
                    return JSONResponse(
                        status_code=HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                        content={
                            "detail": "POST request bodies must use application/json."
                        },
                    )
        return await call_next(request)

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        html = (static_dir / "launcher.html").read_text(encoding="utf-8")
        return html.replace("__RETRIEVER_CONTROL_TOKEN__", json.dumps(control_token))

    @app.get("/api/scenes")
    async def list_scenes() -> list[dict[str, Any]]:
        return [scene.to_dict() for scene in scenes]

    @app.get("/api/status")
    async def status() -> dict[str, Any]:
        return await asyncio.to_thread(manager.status)

    @app.post("/api/launch")
    async def launch(request: LaunchRequest) -> dict[str, Any]:
        scene = by_task.get(request.task)
        if scene is None:
            raise HTTPException(status_code=404, detail="Unknown embodied task")
        if not scene.available:
            raise HTTPException(
                status_code=409,
                detail=scene.unavailable_reason or "Dataset is unavailable.",
            )
        try:
            return await asyncio.to_thread(
                manager.launch,
                scene,
                request.episode,
                goal=request.goal,
                planner=request.planner,
                execution_mode=request.execution_mode,
                interface=request.interface,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

    @app.post("/api/stop")
    async def stop() -> dict[str, Any]:
        try:
            return await asyncio.to_thread(manager.stop)
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

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
        launcher_host=args.host,
    )
    print(f"Retriever RoboCasa scenes: http://localhost:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
