"""Optional browser visualization for robosuite-backed environments."""

from __future__ import annotations

from dataclasses import dataclass, replace
from html import escape
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class ReplaySnapshot:
    """Immutable browser-facing view of a demonstration replay."""

    task: str
    episode: int
    status: str = "Loading"
    paused: bool = False
    speed: float = 1.0
    episode_step: int = 0
    total_steps: int = 0
    cycle: int = 0
    progress: float = 0.0
    reward: float = 0.0
    success: bool = False
    action_norm: float = 0.0


class ReplayControls:
    """Thread-safe controls shared by Viser callbacks and Retriever Flows."""

    def __init__(self, *, task: str, episode: int) -> None:
        self._lock = RLock()
        self._snapshot = ReplaySnapshot(task=task, episode=episode)
        self._step_budget = 0
        self._restart_requested = False

    def snapshot(self) -> ReplaySnapshot:
        with self._lock:
            return self._snapshot

    def set_total_steps(self, total_steps: int) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                total_steps=max(0, int(total_steps)),
                status="Ready",
            )

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                paused=paused,
                status="Paused" if paused else "Running",
            )

    def request_step(self) -> None:
        with self._lock:
            self._step_budget += 1
            self._snapshot = replace(
                self._snapshot,
                paused=True,
                status="Stepping",
            )

    def request_restart(self) -> None:
        with self._lock:
            self._restart_requested = True
            self._step_budget = 0
            self._snapshot = replace(
                self._snapshot,
                paused=False,
                episode_step=0,
                progress=0.0,
                reward=0.0,
                success=False,
                action_norm=0.0,
                status="Restarting",
            )

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("Replay speed must be positive")
        with self._lock:
            self._snapshot = replace(self._snapshot, speed=float(speed))

    def claim_next_action(self) -> tuple[bool, bool]:
        """Return whether the source may advance and whether it should restart."""

        with self._lock:
            if self._restart_requested:
                self._restart_requested = False
                return True, True
            if not self._snapshot.paused:
                return True, False
            if self._step_budget > 0:
                self._step_budget -= 1
                return True, False
            return False, False

    def update_observation(
        self,
        *,
        episode_step: int,
        cycle: int,
        progress: float,
        reward: float,
        success: bool,
        action_norm: float,
    ) -> None:
        with self._lock:
            status = "Success" if success else "Running"
            if self._snapshot.paused and not success:
                status = "Paused"
            self._snapshot = replace(
                self._snapshot,
                status=status,
                episode_step=int(episode_step),
                cycle=int(cycle),
                progress=min(1.0, max(0.0, float(progress))),
                reward=float(reward),
                success=bool(success),
                action_norm=float(action_norm),
            )

    def mark_complete(self) -> None:
        with self._lock:
            self._snapshot = replace(self._snapshot, status="Complete")


class MjviserBridge:
    """Publish an existing robosuite MuJoCo state through mjviser."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8085,
        label: str = "Retriever simulation",
        controls: ReplayControls | None = None,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.label = label
        self.controls = controls
        self._server: Any | None = None
        self._scene: Any | None = None
        self._gui_lock = RLock()
        self._status_markdown: Any | None = None
        self._graph_html: Any | None = None
        self._progress: Any | None = None

    def start(self, sim: Any) -> None:
        if self._scene is not None:
            return

        try:
            import viser
            from mjviser import ViserMujocoScene
        except ImportError as exc:
            raise RuntimeError(
                "mjviser is not installed. Install the optional simulation dependencies "
                'with `python -m pip install -e ".[robosuite]"`.'
            ) from exc

        model, data = _native_mujoco_state(sim)
        self._server = viser.ViserServer(
            host=self.host,
            port=self.port,
            label=self.label,
        )
        self._scene = ViserMujocoScene(self._server, model, num_envs=1)

        # robosuite uses group 0 for collision proxies and group 1 for visual geoms.
        # Keep collisions available in the Groups tab, but do not overlay them by default.
        self._scene.geom_groups_visible[0] = False
        self._scene._sync_visibilities()
        self._scene.create_visualization_gui()
        if self.controls is not None:
            self._create_retriever_panel(viser)
        self._scene.update_from_mjdata(data)
        self.refresh_controls()

        display_host = (
            "localhost" if self.host in {"0.0.0.0", "127.0.0.1"} else self.host
        )
        print(f"Retriever mjviser: http://{display_host}:{self.port}")

    def update(self, sim: Any) -> None:
        self.start(sim)
        if self._scene is None:
            return
        _, data = _native_mujoco_state(sim)
        self._scene.update_from_mjdata(data)

    def refresh_controls(self) -> None:
        if (
            self.controls is None
            or self._status_markdown is None
            or self._graph_html is None
            or self._progress is None
        ):
            return
        snapshot = self.controls.snapshot()
        status = snapshot.status
        if snapshot.success:
            status = "Success"
        with self._gui_lock:
            self._status_markdown.content = _status_markdown(snapshot, status)
            self._graph_html.content = _graph_html(snapshot, status)
            self._progress.value = snapshot.progress * 100.0

    def close(self) -> None:
        if self._server is not None:
            self._server.stop()
        self._scene = None
        self._server = None
        self._status_markdown = None
        self._graph_html = None
        self._progress = None

    def _create_retriever_panel(self, viser: Any) -> None:
        if self._server is None or self.controls is None:
            return

        panel = self._server.gui.add_panel()
        panel.dock_left()
        panel.set_width(340)

        with panel.add_tab("Run", viser.Icon.PLAYER_PLAY):
            self._status_markdown = self._server.gui.add_markdown("")
            self._progress = self._server.gui.add_progress_bar(
                0.0,
                color="green",
            )
            pause = self._server.gui.add_button(
                "Pause",
                icon=viser.Icon.PLAYER_PAUSE,
                hint="Pause before the next recorded action.",
            )
            resume = self._server.gui.add_button(
                "Resume",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Continue replaying recorded actions.",
            )
            step = self._server.gui.add_button(
                "Step",
                icon=viser.Icon.PLAYER_TRACK_NEXT,
                hint="Advance exactly one recorded action, then stay paused.",
            )
            restart = self._server.gui.add_button(
                "Restart episode",
                icon=viser.Icon.REFRESH,
                hint="Reset the simulator and replay this episode from step zero.",
            )
            speed = self._server.gui.add_dropdown(
                "Replay speed",
                ("0.25x", "0.5x", "1x"),
                initial_value="1x",
            )

            @pause.on_click
            def _(_) -> None:
                self.controls.set_paused(True)
                self.refresh_controls()

            @resume.on_click
            def _(_) -> None:
                self.controls.set_paused(False)
                self.refresh_controls()

            @step.on_click
            def _(_) -> None:
                self.controls.request_step()
                self.refresh_controls()

            @restart.on_click
            def _(_) -> None:
                self.controls.request_restart()
                self.refresh_controls()

            @speed.on_update
            def _(_) -> None:
                self.controls.set_speed(float(speed.value.rstrip("x")))
                self.refresh_controls()

        with panel.add_tab("Graph", viser.Icon.GRAPH):
            self._graph_html = self._server.gui.add_html("")


def _status_markdown(snapshot: ReplaySnapshot, status: str) -> str:
    total = snapshot.total_steps
    displayed_step = min(snapshot.episode_step + 1, total) if total else 0
    return (
        f"## Retriever replay\n"
        f"**{snapshot.task}** | episode `{snapshot.episode}`\n\n"
        f"| | |\n|---|---:|\n"
        f"| Status | **{status}** |\n"
        f"| Action | `{displayed_step} / {total}` |\n"
        f"| Cycle | `{snapshot.cycle}` |\n"
        f"| Reward | `{snapshot.reward:.3f}` |\n"
        f"| Action norm | `{snapshot.action_norm:.3f}` |\n"
        f"| Speed | `{snapshot.speed:g}x` |"
    )


def _graph_html(snapshot: ReplaySnapshot, status: str) -> str:
    status_color = "#15803d" if snapshot.success else "#2563eb"
    if snapshot.paused and not snapshot.success:
        status_color = "#a16207"
    task = snapshot.task
    status = escape(status)
    total = snapshot.total_steps
    displayed_step = min(snapshot.episode_step + 1, total) if total else 0
    return (
        '<div style="font-family: Inter, ui-sans-serif, system-ui, sans-serif; '
        'padding: 6px 2px 12px; color: #172033;">'
        '<div style="display:flex; align-items:center; justify-content:space-between; '
        'margin-bottom:12px;">'
        '<strong style="font-size:17px;">Live Retriever Flow</strong>'
        f'<span style="font-size:11px; font-weight:700; color:{status_color}; '
        f'letter-spacing:0.04em;">{status.upper()}</span></div>'
        f"{_flow_node('SOURCE', 'DemoActionSource', '#0e7490', '#ecfeff', f'{task} / action {displayed_step} of {total}')}"
        f"{_flow_edge('RoboCasaAction', 'Latest', '#0e7490')}"
        f"{_flow_node('SIMULATOR', 'RoboCasaSimulator', '#b45309', '#fffbeb', f'MuJoCo / {snapshot.progress:.1%} complete')}"
        f"{_flow_edge('RoboCasaObservation', 'Latest', '#b45309')}"
        f"{_flow_node('TRIGGER', 'ObservationPrinter', '#15803d', '#f0fdf4', f'episode_step / reward {snapshot.reward:.3f}')}"
        '<div style="margin-top:13px; padding-top:10px; border-top:1px solid #d8dee8; '
        'font-size:11px; line-height:1.45; color:#667085;">'
        "Browser controls and graph state share the same thread-safe replay "
        "state used by the Retriever Flows.</div></div>"
    )


def _flow_node(
    kind: str,
    name: str,
    accent: str,
    background: str,
    detail: str,
) -> str:
    return (
        f'<div style="border:1px solid #d8dee8; border-left:4px solid {accent}; '
        f"background:{background}; border-radius:6px; padding:10px 11px; "
        'box-shadow:0 1px 2px rgba(16,24,40,0.06);">'
        '<div style="display:flex; align-items:center; justify-content:space-between; '
        'gap:8px;">'
        f'<strong style="font-size:14px; color:#172033;">{escape(name)}</strong>'
        f'<span style="font-size:9px; font-weight:800; color:{accent}; '
        f'letter-spacing:0.08em;">{escape(kind)}</span></div>'
        f'<div style="font-size:11px; color:#667085; margin-top:4px; '
        f'white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{escape(detail)}</div>'
        "</div>"
    )


def _flow_edge(payload: str, sync: str, accent: str) -> str:
    return (
        '<div style="height:46px; position:relative; margin-left:18px; '
        f'border-left:2px solid {accent};">'
        f'<span style="position:absolute; left:10px; top:9px; background:#ffffff; '
        "border:1px solid #d8dee8; border-radius:4px; padding:3px 6px; "
        f'font:10px ui-monospace, SFMono-Regular, Menlo, monospace; color:#344054;">{escape(payload)}</span>'
        f'<span style="position:absolute; right:2px; top:11px; font-size:10px; '
        f'color:#667085;">{escape(sync)}</span>'
        f'<span style="position:absolute; left:-5px; bottom:-1px; color:{accent}; '
        'font-size:14px; line-height:10px;">&#9660;</span></div>'
    )


def _native_mujoco_state(sim: Any) -> tuple[Any, Any]:
    """Return native MuJoCo objects from robosuite wrappers or direct objects."""

    model = getattr(sim.model, "_model", sim.model)
    data = getattr(sim.data, "_data", sim.data)
    return model, data
