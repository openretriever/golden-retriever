"""Optional browser visualization for robosuite-backed environments."""

from __future__ import annotations

import webbrowser
from contextlib import suppress
from dataclasses import dataclass
from html import escape
from math import radians
from threading import RLock, Thread
from time import monotonic
from typing import Any

from .runtime import ReplayControls, ReplaySnapshot, present_replay


@dataclass(frozen=True)
class _CameraPreset:
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    fov_degrees: float = 55.0


_DEFAULT_CAMERA_PRESETS = {
    "Robot": _CameraPreset(
        position=(1.35, -1.35, 1.55),
        look_at=(0.0, 0.4, 0.72),
        fov_degrees=52.0,
    ),
    "Agent": _CameraPreset(
        position=(0.0, -0.6, 1.85),
        look_at=(0.0, 1.0, 0.8),
    ),
    "Overview": _CameraPreset(
        position=(0.0, -0.6, 5.0),
        look_at=(0.0, 0.5, 0.7),
    ),
}


class MjviserBridge:
    """Publish an existing robosuite MuJoCo state through mjviser."""

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 8085,
        label: str = "Retriever simulation",
        controls: ReplayControls | None = None,
        open_browser: bool = False,
        camera_preset: str = "Agent",
        robot_oriented_camera: bool = True,
    ) -> None:
        self.host = host
        self.port = int(port)
        self.label = label
        self.controls = controls
        self.open_browser = open_browser
        self.robot_oriented_camera = robot_oriented_camera
        self._server: Any | None = None
        self._scene: Any | None = None
        self._lifecycle_lock = RLock()
        self._gui_lock = RLock()
        self._status_markdown: Any | None = None
        self._plan_html: Any | None = None
        self._graph_html: Any | None = None
        self._events_html: Any | None = None
        self._progress: Any | None = None
        self._run_goal_button: Any | None = None
        self._goal_input: Any | None = None
        self._planner_dropdown: Any | None = None
        self._execution_mode_dropdown: Any | None = None
        self._camera_group: Any | None = None
        self._speed_dropdown: Any | None = None
        self._pause_button: Any | None = None
        self._resume_button: Any | None = None
        self._step_button: Any | None = None
        self._restart_button: Any | None = None
        self._last_goal_text = ""
        self._camera_presets = dict(_DEFAULT_CAMERA_PRESETS)
        if camera_preset not in _DEFAULT_CAMERA_PRESETS:
            raise ValueError(f"Unknown camera preset: {camera_preset}")
        self._selected_camera_preset = (
            controls.snapshot().camera_preset if controls is not None else camera_preset
        )
        self._next_control_refresh_at = 0.0

    def start(self, sim: Any) -> None:
        with self._lifecycle_lock:
            if self._scene is not None:
                return

            try:
                import viser
                from mjviser import ViserMujocoScene
            except ImportError as exc:
                raise RuntimeError(
                    "mjviser is not installed. Run the real simulator through "
                    "the locked environment with "
                    "`pixi run --locked -e robocasa ...`."
                ) from exc

            model, data = _native_mujoco_state(sim)
            server: Any | None = None
            self._clear_runtime_state_locked()
            try:
                server = viser.ViserServer(
                    host=self.host,
                    port=self.port,
                    label=self.label,
                )
                scene = ViserMujocoScene(server, model, num_envs=1)
                self._server = server
                self._scene = scene
                robot_body_id = _robot_tracking_body_id(model)
                if robot_body_id is not None:
                    # mjviser currently selects the first movable body, which is
                    # often an invisible RoboCasa target. Anchor tracking to the robot.
                    _set_scene_tracking_body(scene, robot_body_id)
                    if self.robot_oriented_camera:
                        self._camera_presets = _camera_presets_from_robot(
                            data, robot_body_id
                        )

                scene.create_visualization_gui(camera_distance=3.0)
                # robosuite uses group 0 for collision proxies, group 1 for
                # visual geoms, and group 2 for debug targets. The latter can
                # include an end-effector target initialized below the floor.
                for group_id in (0, 2):
                    if len(scene.geom_groups_visible) > group_id:
                        scene.geom_groups_visible[group_id] = False
                    if len(scene.site_groups_visible) > group_id:
                        scene.site_groups_visible[group_id] = False
                _sync_scene_visibilities(scene)
                self._register_camera_handler()
                if self.controls is not None:
                    self._create_retriever_panel(viser)
                scene.update_from_mjdata(data)
                self.refresh_controls()
            except BaseException:
                self._clear_runtime_state_locked()
                if server is not None:
                    with suppress(Exception):
                        server.stop()
                raise

        viewer_url = _viewer_url(self.host, self.port)
        print(f"Retriever mjviser: {viewer_url}")
        if self.open_browser:
            try:
                webbrowser.open(viewer_url)
            except (OSError, webbrowser.Error) as exc:
                print(f"Retriever mjviser: could not open a browser ({exc})")

    def update(self, sim: Any) -> None:
        self.start(sim)
        with self._lifecycle_lock:
            if self._scene is None:
                return
            _, data = _native_mujoco_state(sim)
            self._scene.update_from_mjdata(data)
            now = monotonic()
            if now >= self._next_control_refresh_at:
                self.refresh_controls()
                self._next_control_refresh_at = now + 0.1

    def apply_camera_preset(self, name: str) -> None:
        """Apply a named camera preset to every connected mjviser client."""

        with self._lifecycle_lock:
            if name not in self._camera_presets:
                raise ValueError(f"Unknown camera preset: {name}")
            if self._server is None:
                raise RuntimeError("mjviser is not running")
            self._selected_camera_preset = name
            preset = self._camera_presets[name]
            for client in tuple(self._server.get_clients().values()):
                _apply_camera_preset(client, preset)
            if self.controls is not None:
                self.controls.set_camera_preset(name)

    def refresh_controls(self) -> None:
        with self._lifecycle_lock:
            if (
                self.controls is None
                or self._status_markdown is None
                or self._plan_html is None
                or self._graph_html is None
                or self._events_html is None
                or self._progress is None
            ):
                return
            snapshot = self.controls.snapshot()
            status = present_replay(snapshot).status
            with self._gui_lock:
                self._status_markdown.content = _status_markdown(snapshot, status)
                self._plan_html.content = _plan_html(snapshot)
                self._graph_html.content = _graph_html(snapshot, status)
                self._events_html.content = _events_html(snapshot)
                self._progress.value = snapshot.progress * 100.0
                _set_gui_value(self._planner_dropdown, snapshot.planner)
                _set_gui_value(
                    self._execution_mode_dropdown,
                    snapshot.execution_mode,
                )
                _set_gui_value(self._camera_group, snapshot.camera_preset)
                _set_gui_value(self._speed_dropdown, f"{snapshot.speed:g}x")
                if snapshot.goal_text != self._last_goal_text:
                    _set_gui_value(self._goal_input, snapshot.goal_text)
                    self._last_goal_text = snapshot.goal_text
                _set_gui_disabled(self._run_goal_button, snapshot.planning)
                terminal = present_replay(snapshot).terminal
                _set_gui_disabled(
                    self._pause_button,
                    snapshot.planning or terminal or snapshot.paused,
                )
                _set_gui_disabled(
                    self._resume_button,
                    snapshot.planning or terminal or not snapshot.paused,
                )
                _set_gui_disabled(
                    self._step_button,
                    snapshot.planning or terminal or not snapshot.paused,
                )
                _set_gui_disabled(self._restart_button, snapshot.planning)

    def close(self) -> None:
        if self.controls is not None:
            self.controls.cancel_pending_goals()
        with self._lifecycle_lock:
            server = self._server
            self._clear_runtime_state_locked()
        if server is not None:
            server.stop()

    def _clear_runtime_state_locked(self) -> None:
        self._scene = None
        self._server = None
        self._status_markdown = None
        self._plan_html = None
        self._graph_html = None
        self._events_html = None
        self._progress = None
        self._run_goal_button = None
        self._goal_input = None
        self._planner_dropdown = None
        self._execution_mode_dropdown = None
        self._camera_group = None
        self._speed_dropdown = None
        self._pause_button = None
        self._resume_button = None
        self._step_button = None
        self._restart_button = None
        self._last_goal_text = ""
        self._camera_presets = dict(_DEFAULT_CAMERA_PRESETS)
        self._next_control_refresh_at = 0.0

    def _create_retriever_panel(self, viser: Any) -> None:
        if self._server is None or self.controls is None:
            return

        panel = self._server.gui.add_panel()
        panel.dock_left()
        panel.set_width(340)

        with panel.add_tab("Run", viser.Icon.PLAYER_PLAY):
            initial = self.controls.snapshot()
            goal = self._server.gui.add_text(
                "Goal",
                initial.goal_text or f"Run {initial.task}",
                hint="Plan an allow-listed skill sequence for the selected task.",
            )
            planner = self._server.gui.add_dropdown(
                "Planner",
                ("offline", "gemini"),
                initial_value=initial.planner,
            )
            execution_mode = self._server.gui.add_dropdown(
                "Execution mode",
                ("demonstration", "live_planning"),
                initial_value=initial.execution_mode,
                hint=(
                    "Live planning generates the visible skill plan now; both "
                    "modes replay recorded RoboCasa data, then verify the task."
                ),
            )
            run_goal = self._server.gui.add_button(
                "Run goal",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Validate the plan, restart the episode, and execute it.",
            )
            self._status_markdown = self._server.gui.add_markdown("")
            camera = self._server.gui.add_button_group(
                "Camera",
                tuple(self._camera_presets),
                hint="Switch between third-person, agent, and overview cameras.",
            )
            camera.value = self._selected_camera_preset
            self._progress = self._server.gui.add_progress_bar(
                0.0,
                color="green",
            )
            pause = self._server.gui.add_button(
                "Pause",
                icon=viser.Icon.PLAYER_PAUSE,
                hint="Pause before the next recorded replay step.",
            )
            resume = self._server.gui.add_button(
                "Resume",
                icon=viser.Icon.PLAYER_PLAY,
                hint="Continue replaying recorded data.",
            )
            step = self._server.gui.add_button(
                "Step",
                icon=viser.Icon.PLAYER_TRACK_NEXT,
                hint="Advance exactly one recorded replay step, then stay paused.",
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
            self._run_goal_button = run_goal
            self._goal_input = goal
            self._planner_dropdown = planner
            self._execution_mode_dropdown = execution_mode
            self._camera_group = camera
            self._speed_dropdown = speed
            self._pause_button = pause
            self._resume_button = resume
            self._step_button = step
            self._restart_button = restart
            self._last_goal_text = initial.goal_text

            @run_goal.on_click
            def _(_) -> None:
                text = goal.value
                planner_name = planner.value
                mode = execution_mode.value
                run_goal.hint = "Planning..."
                _set_gui_disabled(run_goal, True)

                def submit() -> None:
                    try:
                        self.controls.submit_goal(
                            text,
                            planner_name,
                            mode,
                            timeout=30.0,
                        )
                        run_goal.hint = (
                            "Plan accepted; restarting the selected episode."
                        )
                    except Exception as exc:  # noqa: BLE001 - GUI command boundary
                        run_goal.hint = str(exc)
                    finally:
                        self.refresh_controls()

                Thread(
                    target=submit,
                    name="retriever-viser-goal",
                    daemon=True,
                ).start()

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

            @camera.on_click
            def _(_) -> None:
                self.apply_camera_preset(camera.value)

        with panel.add_tab("Plan", viser.Icon.TIMELINE):
            self._plan_html = self._server.gui.add_html("")

        with panel.add_tab("Graph", viser.Icon.GRAPH):
            self._graph_html = self._server.gui.add_html("")

        with panel.add_tab("Events", viser.Icon.ACTIVITY):
            self._events_html = self._server.gui.add_html("")

    def _register_camera_handler(self) -> None:
        if self._server is None:
            return

        @self._server.on_client_connect
        def _(client) -> None:
            with self._lifecycle_lock:
                if self._server is None:
                    return
                name = self._selected_camera_preset
                preset = self._camera_presets.get(name, self._camera_presets["Agent"])
                _apply_camera_preset(client, preset)


def _set_gui_value(handle: Any | None, value: Any) -> None:
    if handle is None or getattr(handle, "value", None) == value:
        return
    with suppress(Exception):
        handle.value = value


def _set_gui_disabled(handle: Any | None, disabled: bool) -> None:
    if handle is None:
        return
    with suppress(Exception):
        handle.disabled = disabled


def _status_markdown(snapshot: ReplaySnapshot, status: str) -> str:
    total = snapshot.total_steps
    displayed_step = min(snapshot.episode_step + 1, total) if total else 0
    return (
        f"## Retriever console\n"
        f"**{snapshot.task}** | episode `{snapshot.episode}` | "
        f"`{snapshot.planner}` | `{snapshot.execution_mode}`\n\n"
        f"| | |\n|---|---:|\n"
        f"| Status | **{status}** |\n"
        f"| Action | `{displayed_step} / {total}` |\n"
        f"| Cycle | `{snapshot.cycle}` |\n"
        f"| Reward | `{snapshot.reward:.3f}` |\n"
        f"| Dataset action norm | `{snapshot.action_norm:.3f}` |\n"
        f"| Speed | `{snapshot.speed:g}x` |"
    )


def _graph_html(snapshot: ReplaySnapshot, status: str) -> str:
    status_color = "#15803d" if snapshot.success else "#2563eb"
    if status == "Failed":
        status_color = "#b91c1c"
    if snapshot.paused and not snapshot.success:
        status_color = "#a16207"
    task = snapshot.task
    status = escape(status)
    total = snapshot.total_steps
    displayed_step = min(snapshot.episode_step + 1, total) if total else 0
    active_step = (
        next(
            (
                step
                for step in snapshot.plan.steps
                if step.step_id == snapshot.current_step_id
            ),
            None,
        )
        if snapshot.plan
        else None
    )
    dispatch_detail = (
        f"{active_step.stage_label} / {active_step.label}"
        if active_step is not None
        else "verification failed"
        if status == "Failed"
        else "task verified"
        if snapshot.success
        else "awaiting execution"
    )
    return (
        '<div style="font-family: Inter, ui-sans-serif, system-ui, sans-serif; '
        'padding: 6px 2px 12px; color: #172033;">'
        '<div style="display:flex; align-items:center; justify-content:space-between; '
        'margin-bottom:12px;">'
        '<strong style="font-size:17px;">Retriever pipeline map</strong>'
        f'<span style="font-size:11px; font-weight:700; color:{status_color}; '
        f'letter-spacing:0.04em;">{status.upper()}</span></div>'
        f"{_flow_node('GOAL', 'GoalSource', '#475569', '#f8fafc', snapshot.goal_text or task)}"
        f"{_flow_edge('EmbodiedGoal', 'Trigger', '#475569')}"
        f"{_flow_node('PLAN', 'EmbodiedPlanner', '#7c3aed', '#faf5ff', snapshot.planner)}"
        f"{_flow_edge('SkillPlan', 'Trigger', '#7c3aed')}"
        f"{_flow_node('DISPATCH', 'SkillDispatcher', '#0369a1', '#f0f9ff', dispatch_detail)}"
        f"{_flow_edge('ExecutionState', 'Latest', '#0369a1')}"
        f"{_flow_node('SOURCE', 'DemoActionSource', '#0e7490', '#ecfeff', f'{task} / action {displayed_step} of {total}')}"
        f"{_flow_edge('RoboCasaAction', 'Latest', '#0e7490')}"
        f"{_flow_node('SIMULATOR', 'RoboCasaSimulator', '#b45309', '#fffbeb', f'MuJoCo / {snapshot.progress:.1%} complete')}"
        f"{_flow_edge('RoboCasaObservation', 'Latest', '#b45309')}"
        f"{_flow_node('VERIFY', 'TaskVerifier', '#15803d', '#f0fdf4', f'reward {snapshot.reward:.3f} / success {snapshot.success}')}"
        f"{_flow_edge('ExecutionEvent', 'Trigger', '#15803d')}"
        f"{_flow_node('SINK', 'EventSink', '#be123c', '#fff1f2', f'{len(snapshot.events)} lifecycle events')}"
        '<div style="margin-top:13px; padding-top:10px; border-top:1px solid #d8dee8; '
        'font-size:11px; line-height:1.45; color:#667085;">'
        "Browser controls and graph state share the same thread-safe replay "
        "state used by the Retriever Flows.</div></div>"
    )


def _plan_html(snapshot: ReplaySnapshot) -> str:
    plan = snapshot.plan
    if plan is None:
        return _empty_panel("No plan yet", "Submit a goal from the Run tab.")
    stages: list[tuple[str, str, list[tuple[int, Any]]]] = []
    for index, step in enumerate(plan.steps, start=1):
        if not stages or stages[-1][0] != step.stage_id:
            stages.append((step.stage_id, step.stage_label, []))
        stages[-1][2].append((index, step))

    completed = {
        event.step_id
        for event in snapshot.events
        if event.status == "completed" and event.step_id
    }
    failed = {
        event.step_id
        for event in snapshot.events
        if event.status == "failed" and event.step_id
    }
    stage_rows: list[str] = []
    for stage_index, (_stage_id, stage_label, stage_steps) in enumerate(
        stages, start=1
    ):
        rows: list[str] = []
        states: list[str] = []
        for index, step in stage_steps:
            if step.step_id in failed:
                state, color, symbol = "Failed", "#b91c1c", "!"
            elif step.step_id == snapshot.current_step_id:
                state, color, symbol = "Current", "#2563eb", "&#9654;"
            elif step.step_id in completed:
                state, color, symbol = "Passed", "#15803d", "&#10003;"
            else:
                state, color, symbol = "Pending", "#64748b", str(index)
            states.append(state)
            rows.append(
                '<div style="display:grid;grid-template-columns:24px 1fr auto;gap:8px;'
                'align-items:center;padding:7px 3px 7px 10px;border-left:1px solid #d8dee8;">'
                f'<span style="width:21px;height:21px;border:1px solid {color};border-radius:50%;'
                f'color:{color};display:grid;place-items:center;font-size:10px;font-weight:800;">{symbol}</span>'
                f'<div><strong style="font-size:12px;color:#172033;">{escape(step.label)}</strong>'
                f'<div style="font-size:9px;color:#667085;margin-top:2px;">{escape(step.skill)} | {escape(step.lane)}</div></div>'
                f'<span style="font-size:8px;font-weight:800;color:{color};text-transform:uppercase;">{state}</span></div>'
            )
        if "Failed" in states:
            stage_state, stage_color, stage_symbol = "Failed", "#b91c1c", "!"
        elif all(state == "Passed" for state in states):
            stage_state, stage_color, stage_symbol = "Passed", "#15803d", "&#10003;"
        elif "Current" in states:
            stage_state, stage_color, stage_symbol = (
                "Current",
                "#2563eb",
                str(stage_index),
            )
        else:
            stage_state, stage_color, stage_symbol = (
                "Queued",
                "#64748b",
                str(stage_index),
            )
        stage_rows.append(
            '<section style="margin-bottom:14px;">'
            '<div style="display:grid;grid-template-columns:25px 1fr auto;gap:8px;'
            'align-items:center;margin-bottom:4px;">'
            f'<span style="width:22px;height:22px;border:1px solid {stage_color};border-radius:4px;'
            f'color:{stage_color};display:grid;place-items:center;font-size:10px;font-weight:800;">{stage_symbol}</span>'
            f'<strong style="font-size:12px;color:#172033;">{escape(stage_label)}</strong>'
            f'<span style="font-size:8px;font-weight:800;color:{stage_color};text-transform:uppercase;">{stage_state}</span>'
            "</div>" + "".join(rows) + "</section>"
        )
    return (
        '<div style="font-family:Inter,ui-sans-serif,system-ui,sans-serif;padding:6px 2px 12px;color:#172033;">'
        '<div style="font-size:17px;font-weight:750;margin-bottom:3px;">Skill timeline</div>'
        f'<div style="font-size:11px;color:#667085;margin-bottom:13px;">{escape(plan.source)} plan | {len(stages)} subplans | {len(plan.steps)} skills</div>'
        + "".join(stage_rows)
        + "</div>"
    )


def _events_html(snapshot: ReplaySnapshot) -> str:
    if not snapshot.events:
        return _empty_panel("No events yet", "Execution events appear after dispatch.")
    colors = {
        "completed": "#15803d",
        "running": "#2563eb",
        "failed": "#b91c1c",
        "verified": "#047857",
    }
    rows = []
    for event in reversed(snapshot.events[-24:]):
        color = colors.get(event.status, "#64748b")
        rows.append(
            '<div style="display:grid;grid-template-columns:45px 9px 1fr;gap:8px;'
            'align-items:start;padding:8px 2px;border-bottom:1px solid #e5e7eb;">'
            f'<span style="font:10px ui-monospace,SFMono-Regular,Menlo,monospace;color:#667085;">+{event.elapsed_seconds:.1f}s</span>'
            f'<span style="width:8px;height:8px;border-radius:50%;background:{color};margin-top:3px;"></span>'
            f'<div><strong style="font-size:11px;color:{color};text-transform:uppercase;">{escape(event.status)}</strong>'
            f'<div style="font-size:12px;color:#344054;margin-top:2px;line-height:1.35;">{escape(event.message)}</div></div></div>'
        )
    return (
        '<div style="font-family:Inter,ui-sans-serif,system-ui,sans-serif;padding:6px 2px 12px;color:#172033;">'
        '<div style="font-size:17px;font-weight:750;margin-bottom:10px;">Execution events</div>'
        + "".join(rows)
        + "</div>"
    )


def _empty_panel(title: str, detail: str) -> str:
    return (
        '<div style="font-family:Inter,ui-sans-serif,system-ui,sans-serif;'
        'padding:18px 4px;color:#667085;">'
        f'<strong style="display:block;color:#344054;margin-bottom:5px;">'
        f"{escape(title)}</strong>{escape(detail)}</div>"
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


def _set_scene_tracking_body(scene: Any, body_id: int) -> None:
    """Bridge current and future mjviser tracking APIs in one place."""

    setter = getattr(scene, "set_tracked_body", None)
    if callable(setter):
        setter(body_id)
        return
    if hasattr(scene, "tracked_body_id"):
        scene.tracked_body_id = body_id
        return
    if hasattr(scene, "_tracked_body_id"):
        scene._tracked_body_id = body_id
        return
    raise RuntimeError("This mjviser version does not expose body tracking")


def _sync_scene_visibilities(scene: Any) -> None:
    """Apply geom visibility through whichever mjviser API is available."""

    sync = getattr(scene, "sync_visibilities", None)
    if not callable(sync):
        sync = getattr(scene, "_sync_visibilities", None)
    if not callable(sync):
        raise TypeError("This mjviser version cannot synchronize geom groups")
    sync()


def _viewer_url(host: str, port: int) -> str:
    """Return a browser-safe local URL for IPv4, IPv6, and wildcard binds."""

    display_host = (
        "localhost" if host in {"0.0.0.0", "127.0.0.1", "::", "::1"} else host
    )
    if ":" in display_host and not display_host.startswith("["):
        display_host = f"[{display_host}]"
    return f"http://{display_host}:{port}"


def _robot_tracking_body_id(model: Any) -> int | None:
    """Choose a stable robot body instead of an invisible control target."""

    candidates: list[tuple[int, int]] = []
    for body_id in range(int(getattr(model, "nbody", 0))):
        name = str(getattr(model.body(body_id), "name", "") or "").lower()
        if name.startswith("mobilebase") and name.endswith("_base"):
            candidates.append((0, body_id))
        elif name.startswith("robot") and name.endswith("_link0"):
            candidates.append((1, body_id))
        elif name.startswith("robot") and name.endswith("_base"):
            candidates.append((2, body_id))
    return min(candidates)[1] if candidates else None


def _apply_camera_preset(client: Any, preset: _CameraPreset) -> None:
    with client.atomic():
        client.camera.position = preset.position
        client.camera.look_at = preset.look_at
        client.camera.up_direction = (0.0, 0.0, 1.0)
        client.camera.fov = radians(preset.fov_degrees)
        client.camera.min_orbit_distance = 0.05
        client.camera.max_orbit_distance = 20.0


def _camera_presets_from_robot(
    data: Any,
    body_id: int,
) -> dict[str, _CameraPreset]:
    """Orient camera presets from the robot base into its workspace."""

    try:
        matrix = data.xmat[body_id].reshape(3, 3)
        forward = tuple(float(matrix[index, 0]) for index in range(3))
    except (AttributeError, IndexError, TypeError, ValueError):
        return _DEFAULT_CAMERA_PRESETS

    try:
        origin = tuple(float(value) for value in data.xpos[body_id])
    except (AttributeError, IndexError, TypeError, ValueError):
        origin = (0.0, 0.0, 0.0)

    up = (0.0, 0.0, 1.0)
    return {
        "Robot": _CameraPreset(
            position=_vector_sum(
                origin,
                _scaled(forward, -1.0),
                _scaled(up, 2.6),
            ),
            look_at=_vector_sum(
                origin,
                _scaled(forward, 0.3),
                _scaled(up, 1.0),
            ),
            fov_degrees=65.0,
        ),
        "Agent": _CameraPreset(
            position=_vector_sum(
                origin,
                _scaled(forward, -0.6),
                _scaled(up, 1.85),
            ),
            look_at=_vector_sum(
                origin,
                _scaled(forward, 1.0),
                _scaled(up, 0.8),
            ),
            fov_degrees=60.0,
        ),
        "Overview": _CameraPreset(
            position=_vector_sum(
                origin,
                _scaled(forward, -0.6),
                _scaled(up, 5.0),
            ),
            look_at=_vector_sum(
                origin,
                _scaled(forward, 0.5),
                _scaled(up, 0.7),
            ),
        ),
    }


def _scaled(
    vector: tuple[float, float, float], scale: float
) -> tuple[float, float, float]:
    return (vector[0] * scale, vector[1] * scale, vector[2] * scale)


def _vector_sum(*vectors: tuple[float, float, float]) -> tuple[float, float, float]:
    return (
        sum(vector[0] for vector in vectors),
        sum(vector[1] for vector in vectors),
        sum(vector[2] for vector in vectors),
    )
