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

from .mjviser_bridge import ReplayControls, ReplaySnapshot


def snapshot_payload(snapshot: ReplaySnapshot) -> dict[str, Any]:
    """Return a JSON-safe snapshot for browser and remote console clients."""

    return asdict(snapshot)


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
    ) -> None:
        self.controls = controls
        self.viewer_url = viewer_url.rstrip("/")
        self.host = host
        self.port = int(port)
        self.camera_handler = camera_handler
        self.open_browser = open_browser
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
                    self._send_json(
                        HTTPStatus.OK,
                        {
                            "viewer_url": console.viewer_url,
                            "task": console.controls.snapshot().task,
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
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    raw = self.rfile.read(length) if length else b"{}"
                    payload = json.loads(raw or b"{}")
                    if not isinstance(payload, dict):
                        raise TypeError("Request body must be a JSON object")
                    result = console.handle_command(self.path, payload)
                except (
                    TypeError,
                    ValueError,
                    RuntimeError,
                    json.JSONDecodeError,
                ) as exc:
                    self._send_json(HTTPStatus.BAD_REQUEST, {"error": str(exc)})
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
            plan = self.controls.submit_goal(text, planner, execution_mode)
        elif path == "/api/camera":
            preset = str(payload.get("preset", ""))
            if preset not in {"Robot", "Agent", "Overview"}:
                raise ValueError(f"Unknown camera preset: {preset}")
            if self.camera_handler is None:
                raise RuntimeError("No simulator camera is connected")
            self.camera_handler(preset)
        else:
            raise ValueError(f"Unknown console command: {path}")
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
            "The skill plan was generated now; execution uses the verified trajectory."
        )
    return "Replaying the pinned demonstrated plan and verified trajectory."
