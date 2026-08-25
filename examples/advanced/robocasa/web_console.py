"""Renderer-neutral web console for Retriever embodied demonstrations."""

from __future__ import annotations

import json
import webbrowser
from collections.abc import Callable
from dataclasses import asdict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import Any
from urllib.parse import urlsplit

from .embodied import EXECUTION_MODES
from .runtime import (
    PlannerBusyError,
    PlannerCancelledError,
    PlannerTimeoutError,
    PlannerUnavailableError,
    ReplayControls,
    ReplaySnapshot,
    present_replay,
)


class _CommandNotFoundError(ValueError):
    """Raised when the requested console command does not exist."""


class _CommandConflictError(RuntimeError):
    """Raised when a valid command conflicts with newer runtime state."""


class _ServiceUnavailableError(RuntimeError):
    """Raised when a command requires a disconnected local service."""


class _PlannerError(RuntimeError):
    """Raised when a configured planner fails to produce a plan."""


class _PlannerTimeoutError(_PlannerError):
    """Raised when a configured planner exceeds its response deadline."""


def snapshot_payload(snapshot: ReplaySnapshot) -> dict[str, Any]:
    """Return a JSON-safe snapshot for browser and remote console clients."""

    payload = asdict(snapshot)
    presentation = present_replay(snapshot)
    payload["status"] = presentation.status
    payload["presentation"] = asdict(presentation)
    payload["terminal"] = presentation.terminal
    return payload


class RetrieverWebConsole:
    """Serve Retriever controls around a replaceable simulator viewport."""

    def __init__(
        self,
        controls: ReplayControls,
        viewer_url: str,
        *,
        host: str = "127.0.0.1",
        port: int = 8086,
        camera_handler: Callable[[str], None] | None = None,
        open_browser: bool = False,
        launch_id: str = "",
        planner_timeout: float = 30.0,
    ) -> None:
        self.controls = controls
        self.viewer_url = viewer_url.rstrip("/")
        self.host = host
        self.port = int(port)
        self.camera_handler = camera_handler
        self.open_browser = open_browser
        self.launch_id = launch_id
        self.planner_timeout = float(planner_timeout)
        self._server: ThreadingHTTPServer | None = None
        self._thread: Thread | None = None

    @property
    def url(self) -> str:
        display_host = (
            "localhost" if self.host in {"0.0.0.0", "127.0.0.1"} else self.host
        )
        return f"http://{display_host}:{self.port}"

    def start(self) -> None:
        if self._server is not None:
            return

        console = self

        class RequestHandler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:
                if self.path == "/":
                    html = (
                        Path(__file__)
                        .with_name("static")
                        .joinpath("console.html")
                        .read_bytes()
                    )
                    self._send_bytes(HTTPStatus.OK, "text/html; charset=utf-8", html)
                    return
                if self.path == "/api/config":
                    snapshot = console.controls.snapshot()
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "schema_version": 1,
                            "viewer_url": console.viewer_url,
                            "task": snapshot.task,
                            "episode": snapshot.episode,
                            "launch_id": console.launch_id,
                            "renderer": "mjviser",
                            "camera_preset": snapshot.camera_preset,
                            "capabilities": {
                                "camera_presets": (
                                    ["Robot", "Agent", "Overview"]
                                    if console.camera_handler is not None
                                    else []
                                ),
                                "goal_submission": console.controls.can_submit_goals,
                            },
                        },
                    )
                    return
                if self.path == "/api/state":
                    self._send_json(
                        HTTPStatus.OK,
                        snapshot_payload(console.controls.snapshot()),
                    )
                    return
                self._send_json(HTTPStatus.NOT_FOUND, {"error": "Not found"})

            def do_POST(self) -> None:
                origin = self.headers.get("Origin")
                if origin and not self._is_same_origin(origin):
                    self._send_error(
                        HTTPStatus.FORBIDDEN,
                        "cross_origin_forbidden",
                        "Cross-origin console commands are not allowed",
                    )
                    return
                if self.headers.get_content_type() != "application/json":
                    self._send_error(
                        HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                        "unsupported_media_type",
                        "POST requests require Content-Type: application/json",
                    )
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length else b"{}"
                    payload = json.loads(raw or b"{}")
                    if not isinstance(payload, dict):
                        raise TypeError("Request body must be a JSON object")
                    result = console.handle_command(self.path, payload)
                except _CommandNotFoundError as exc:
                    self._send_error(HTTPStatus.NOT_FOUND, "unknown_command", exc)
                    return
                except _CommandConflictError as exc:
                    self._send_error(HTTPStatus.CONFLICT, "command_conflict", exc)
                    return
                except _PlannerTimeoutError as exc:
                    self._send_error(HTTPStatus.GATEWAY_TIMEOUT, "planner_timeout", exc)
                    return
                except _PlannerError as exc:
                    self._send_error(HTTPStatus.BAD_GATEWAY, "planner_error", exc)
                    return
                except _ServiceUnavailableError as exc:
                    self._send_error(
                        HTTPStatus.SERVICE_UNAVAILABLE,
                        "service_unavailable",
                        exc,
                    )
                    return
                except (TypeError, ValueError, json.JSONDecodeError) as exc:
                    self._send_error(HTTPStatus.BAD_REQUEST, "invalid_request", exc)
                    return
                except Exception as exc:  # pragma: no cover - defensive HTTP boundary
                    self._send_error(
                        HTTPStatus.INTERNAL_SERVER_ERROR,
                        "internal_error",
                        f"Console command failed: {exc}",
                    )
                    return
                self._send_json(HTTPStatus.OK, result)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

            def _send_json(self, status: HTTPStatus, payload: Any) -> None:
                self._send_bytes(
                    status,
                    "application/json; charset=utf-8",
                    json.dumps(payload).encode("utf-8"),
                )

            def _send_error(
                self,
                status: HTTPStatus,
                code: str,
                error: object,
            ) -> None:
                self._send_json(status, {"error": str(error), "code": code})

            def _is_same_origin(self, origin: str) -> bool:
                parsed = urlsplit(origin)
                host = self.headers.get("Host", "")
                return parsed.scheme == "http" and parsed.netloc == host

            def _send_bytes(
                self,
                status: HTTPStatus,
                content_type: str,
                payload: bytes,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(payload)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(payload)

        server = ThreadingHTTPServer((self.host, self.port), RequestHandler)
        server.daemon_threads = True
        self.port = int(server.server_address[1])
        thread = Thread(
            target=server.serve_forever,
            name="retriever-web-console",
            daemon=True,
        )
        try:
            thread.start()
        except Exception:
            server.server_close()
            raise
        self._server = server
        self._thread = thread
        print(f"Retriever embodied console: {self.url}")
        if self.open_browser:
            webbrowser.open(self.url)

    def close(self) -> None:
        self.controls.cancel_pending_goals()
        server = self._server
        thread = self._thread
        if server is not None:
            server.shutdown()
            server.server_close()
        if thread is not None and thread.is_alive():
            thread.join(timeout=2.0)
        self._server = None
        self._thread = None

    def handle_command(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Apply one browser command without coupling it to an HTTP framework."""

        plan = None
        if path == "/api/pause":
            self.controls.set_paused(True)
        elif path == "/api/resume":
            self.controls.set_paused(False)
        elif path == "/api/step":
            self.controls.request_step()
        elif path == "/api/restart":
            if self.controls.snapshot().status.lower() != "restarting":
                self.controls.request_restart()
        elif path == "/api/speed":
            try:
                speed = float(payload["speed"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Replay speed must be a number") from exc
            self.controls.set_speed(speed)
        elif path == "/api/goal":
            text = str(payload.get("text", "")).strip()
            if not text:
                raise ValueError("Goal text is required")
            planner = str(payload.get("planner") or "offline")
            execution_mode = str(payload.get("execution_mode") or "demonstration")
            if execution_mode not in EXECUTION_MODES:
                raise ValueError(f"Unknown execution mode: {execution_mode}")
            try:
                plan = self.controls.submit_goal(
                    text,
                    planner,
                    execution_mode,
                    timeout=self.planner_timeout,
                )
            except PlannerTimeoutError as exc:
                raise _PlannerTimeoutError(str(exc)) from exc
            except (PlannerBusyError, PlannerCancelledError) as exc:
                raise _CommandConflictError(str(exc)) from exc
            except PlannerUnavailableError as exc:
                raise _ServiceUnavailableError(str(exc)) from exc
            except Exception as exc:
                raise _PlannerError(f"Planner failed: {exc}") from exc
        elif path == "/api/camera":
            preset = str(payload.get("preset", ""))
            if preset not in {"Robot", "Agent", "Overview"}:
                raise ValueError(f"Unknown camera preset: {preset}")
            if self.camera_handler is None:
                raise _ServiceUnavailableError("No simulator camera is connected")
            try:
                self.camera_handler(preset)
                self.controls.set_camera_preset(preset)
            except Exception as exc:
                raise _ServiceUnavailableError(
                    f"Simulator camera command failed: {exc}"
                ) from exc
        else:
            raise _CommandNotFoundError(f"Unknown console command: {path}")
        result = {"ok": True, "state": snapshot_payload(self.controls.snapshot())}
        if plan is not None:
            stage_count = len(dict.fromkeys(step.stage_id for step in plan.steps))
            result.update(
                {
                    "reply": (
                        f"Plan ready for {plan.goal.task}: {stage_count} subplans and "
                        f"{len(plan.steps)} allow-listed skills from the {plan.source} "
                        f"planner. {_execution_reply(plan.goal.execution_mode)} "
                        f"Restarting episode {plan.goal.episode}."
                    ),
                    "plan_summary": {
                        "task": plan.goal.task,
                        "source": plan.source,
                        "stages": stage_count,
                        "steps": len(plan.steps),
                        "execution_mode": plan.goal.execution_mode,
                    },
                }
            )
        return result


def _execution_reply(execution_mode: str) -> str:
    if execution_mode == "live_planning":
        return (
            "The skill plan was generated now; execution uses recorded replay data "
            "and reports RoboCasa task verification."
        )
    return "Replaying recorded RoboCasa data and reporting task verification."
