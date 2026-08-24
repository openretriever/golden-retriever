"""Optional browser visualization for robosuite-backed environments."""

from __future__ import annotations

import webbrowser
from dataclasses import dataclass, replace
from html import escape
from math import radians
from threading import RLock
from time import monotonic
from typing import Any

from .embodied import EmbodiedGoal, ExecutionEvent, SkillPlan


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
    goal_text: str = ""
    planner: str = "offline"
    execution_mode: str = "demonstration"
    plan: SkillPlan | None = None
    current_step_id: str = ""
    events: tuple[ExecutionEvent, ...] = ()


class ReplayControls:
    """Thread-safe controls shared by Viser callbacks and Retriever Flows."""

    def __init__(self, *, task: str, episode: int) -> None:
        self._lock = RLock()
        self._snapshot = ReplaySnapshot(task=task, episode=episode)
        self._step_budget = 0
        self._restart_requested = False
        self._started_at = monotonic()
        self._event_sequence = 0
        self._goal_handler: Any | None = None
        self._goal_generation = 0
        self._source_exhausted = False

    def set_goal_handler(self, handler: Any) -> None:
        """Install the planner callback used by the browser goal composer."""

        with self._lock:
            self._goal_handler = handler

    def submit_goal(
        self,
        text: str,
        planner: str | None = None,
        execution_mode: str | None = None,
    ) -> SkillPlan:
        with self._lock:
            handler = self._goal_handler
            snapshot = self._snapshot
            self._goal_generation += 1
            generation = self._goal_generation
        if handler is None:
            raise RuntimeError("No embodied planner is connected")
        goal = EmbodiedGoal(
            text=text.strip(),
            task=snapshot.task,
            episode=snapshot.episode,
            planner=planner or snapshot.planner,
            execution_mode=execution_mode or snapshot.execution_mode,
        )
        plan = handler(goal)
        plan.validate()
        with self._lock:
            if generation != self._goal_generation:
                raise RuntimeError("Goal was superseded by a newer request")
            self._configure_execution_locked(goal, plan)
            self._request_restart_locked()
        return plan

    def configure_execution(self, goal: EmbodiedGoal, plan: SkillPlan) -> None:
        plan.validate()
        with self._lock:
            self._goal_generation += 1
            self._configure_execution_locked(goal, plan)

    def cancel_pending_goals(self) -> None:
        """Prevent an in-flight planner response from mutating this run."""

        with self._lock:
            self._goal_generation += 1

    def snapshot(self) -> ReplaySnapshot:
        with self._lock:
            return self._snapshot

    def set_total_steps(self, total_steps: int) -> None:
        with self._lock:
            self._source_exhausted = False
            self._snapshot = replace(
                self._snapshot,
                total_steps=max(0, int(total_steps)),
                status="Ready",
            )

    def set_paused(self, paused: bool) -> None:
        with self._lock:
            if self._is_terminal_locked():
                return
            self._snapshot = replace(
                self._snapshot,
                paused=paused,
                status="Paused" if paused else "Running",
            )

    def request_step(self) -> None:
        with self._lock:
            if self._is_terminal_locked():
                return
            self._step_budget += 1
            self._snapshot = replace(
                self._snapshot,
                paused=True,
                status="Stepping",
            )

    def request_restart(self) -> None:
        with self._lock:
            self._request_restart_locked()

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
            if int(cycle) < self._snapshot.cycle:
                return
            success = self._snapshot.success or bool(success)
            final_observation = (
                self._snapshot.total_steps > 0
                and int(episode_step) >= self._snapshot.total_steps - 1
            )
            if success:
                status = "Success"
            elif self._source_exhausted:
                status = "Failed" if final_observation else "Verifying"
            elif self._snapshot.paused and not success:
                status = "Paused"
            else:
                status = "Running"
            self._snapshot = replace(
                self._snapshot,
                status=status,
                episode_step=int(episode_step),
                cycle=int(cycle),
                progress=min(1.0, max(0.0, float(progress))),
                reward=float(reward),
                success=success,
                action_norm=float(action_norm),
            )
            self._update_plan_events(
                self._snapshot.progress,
                success,
                terminal_failure=self._source_exhausted and final_observation,
            )

    def mark_complete(self) -> None:
        with self._lock:
            self._source_exhausted = True
            verified = self._snapshot.success or any(
                event.kind == "verification" and event.status == "verified"
                for event in self._snapshot.events
            )
            final_observation = (
                self._snapshot.total_steps > 0
                and self._snapshot.episode_step >= self._snapshot.total_steps - 1
            )
            status = (
                "Success"
                if verified
                else "Failed"
                if final_observation
                else "Verifying"
            )
            self._snapshot = replace(
                self._snapshot,
                status=status,
                success=verified,
            )
            self._update_plan_events(
                self._snapshot.progress,
                verified,
                terminal_failure=final_observation and not verified,
            )

    def _is_terminal_locked(self) -> bool:
        return self._snapshot.success or self._source_exhausted

    def _configure_execution_locked(
        self,
        goal: EmbodiedGoal,
        plan: SkillPlan,
    ) -> None:
        self._started_at = monotonic()
        self._event_sequence = 0
        self._source_exhausted = False
        self._snapshot = replace(
            self._snapshot,
            task=goal.task,
            episode=goal.episode,
            goal_text=goal.text,
            planner=plan.source,
            execution_mode=goal.execution_mode,
            plan=plan,
            current_step_id=plan.steps[0].step_id,
            events=(),
            status="Ready",
        )
        self._append_event(
            kind="dispatch",
            status="completed",
            step_id="",
            message=(
                f"Plan accepted from {plan.source} planner "
                f"({goal.execution_mode.replace('_', ' ')})"
            ),
        )
        self._append_event(
            kind="skill",
            status="running",
            step_id=plan.steps[0].step_id,
            message=plan.steps[0].label,
        )

    def _request_restart_locked(self) -> None:
        if not self._restart_requested:
            next_cycle = self._snapshot.cycle + 1
        else:
            next_cycle = self._snapshot.cycle
        self._restart_requested = True
        self._source_exhausted = False
        self._step_budget = 0
        self._snapshot = replace(
            self._snapshot,
            paused=False,
            episode_step=0,
            cycle=next_cycle,
            progress=0.0,
            reward=0.0,
            success=False,
            action_norm=0.0,
            status="Restarting",
        )
        if self._snapshot.plan is not None:
            plan = self._snapshot.plan
            self._started_at = monotonic()
            self._event_sequence = 0
            self._snapshot = replace(
                self._snapshot,
                current_step_id=plan.steps[0].step_id,
                events=(),
            )
            self._append_event(
                kind="dispatch",
                status="completed",
                step_id="",
                message="Episode restarted",
            )
            self._append_event(
                kind="skill",
                status="running",
                step_id=plan.steps[0].step_id,
                message=plan.steps[0].label,
            )

    def _update_plan_events(
        self,
        progress: float,
        success: bool,
        *,
        terminal_failure: bool = False,
    ) -> None:
        plan = self._snapshot.plan
        if plan is None:
            return
        if success:
            self._snapshot = replace(
                self._snapshot,
                events=tuple(
                    event
                    for event in self._snapshot.events
                    if event.status != "failed"
                ),
            )
        existing = {(event.step_id, event.status) for event in self._snapshot.events}
        verified = ("", "verified") in existing
        terminal_failure = terminal_failure and not success and not verified
        for step in plan.steps:
            is_verification = step.skill == "verify"
            complete = success or (
                not is_verification and progress >= step.end_fraction
            )
            if complete and (step.step_id, "completed") not in existing:
                self._append_event(
                    kind="skill",
                    status="completed",
                    step_id=step.step_id,
                    message=step.label,
                )
                existing.add((step.step_id, "completed"))

        active = plan.step_at(progress)
        if terminal_failure:
            verification = plan.steps[-1]
            if (verification.step_id, "failed") not in existing:
                self._append_event(
                    kind="skill",
                    status="failed",
                    step_id=verification.step_id,
                    message=verification.label,
                )
                existing.add((verification.step_id, "failed"))
        elif not success and (active.step_id, "running") not in existing:
            self._append_event(
                kind="skill",
                status="running",
                step_id=active.step_id,
                message=active.label,
            )
        self._snapshot = replace(
            self._snapshot,
            current_step_id="" if success or terminal_failure else active.step_id,
        )
        if success and ("", "verified") not in existing:
            self._append_event(
                kind="verification",
                status="verified",
                step_id="",
                message="RoboCasa task success signal verified",
            )
        if terminal_failure and ("", "failed") not in existing:
            self._append_event(
                kind="verification",
                status="failed",
                step_id="",
                message="RoboCasa task success signal was not observed",
            )

    def _append_event(
        self,
        *,
        kind: str,
        status: str,
        step_id: str,
        message: str,
    ) -> None:
        self._event_sequence += 1
        event = ExecutionEvent(
            sequence=self._event_sequence,
            kind=kind,
            status=status,
            step_id=step_id,
            message=message,
            elapsed_seconds=max(0.0, monotonic() - self._started_at),
        )
        self._snapshot = replace(
            self._snapshot,
            events=(*self._snapshot.events[-63:], event),
        )


@dataclass(frozen=True)
class _CameraPreset:
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    fov_degrees: float = 55.0


_DEFAULT_CAMERA_PRESETS = {
    "Robot": _CameraPreset(
        position=(0.0, -1.0, 2.6),
        look_at=(0.0, 0.3, 1.0),
        fov_degrees=65.0,
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
    ) -> None:
        self.host = host
        self.port = int(port)
        self.label = label
        self.controls = controls
        self.open_browser = open_browser
        self._server: Any | None = None
        self._scene: Any | None = None
        self._gui_lock = RLock()
        self._status_markdown: Any | None = None
        self._plan_html: Any | None = None
        self._graph_html: Any | None = None
        self._events_html: Any | None = None
        self._progress: Any | None = None
        self._camera_presets = _DEFAULT_CAMERA_PRESETS
        self._next_control_refresh_at = 0.0

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
        robot_body_id = _robot_tracking_body_id(model)
        if robot_body_id is not None:
            # mjviser currently selects the first movable body, which is often an
            # invisible RoboCasa target. Anchor its tracking offset to the robot.
            self._scene._tracked_body_id = robot_body_id
            self._camera_presets = _camera_presets_from_robot(data, robot_body_id)

        # robosuite uses group 0 for collision proxies and group 1 for visual geoms.
        # Keep collisions available in the Groups tab, but do not overlay them by default.
        self._scene.geom_groups_visible[0] = False
        self._scene._sync_visibilities()
        self._scene.create_visualization_gui(camera_distance=3.0)
        self._register_camera_handler()
        if self.controls is not None:
            self._create_retriever_panel(viser)
        self._scene.update_from_mjdata(data)
        self.refresh_controls()

        display_host = (
            "localhost" if self.host in {"0.0.0.0", "127.0.0.1"} else self.host
        )
        viewer_url = f"http://{display_host}:{self.port}"
        print(f"Retriever mjviser: {viewer_url}")
        if self.open_browser:
            webbrowser.open(viewer_url)

    def update(self, sim: Any) -> None:
        self.start(sim)
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

        if name not in self._camera_presets:
            raise ValueError(f"Unknown camera preset: {name}")
        if self._server is None:
            raise RuntimeError("mjviser is not running")
        preset = self._camera_presets[name]
        for client in tuple(self._server.get_clients().values()):
            _apply_camera_preset(client, preset)

    def refresh_controls(self) -> None:
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
        status = snapshot.status
        if snapshot.success:
            status = "Success"
        with self._gui_lock:
            self._status_markdown.content = _status_markdown(snapshot, status)
            self._plan_html.content = _plan_html(snapshot)
            self._graph_html.content = _graph_html(snapshot, status)
            self._events_html.content = _events_html(snapshot)
            self._progress.value = snapshot.progress * 100.0

    def close(self) -> None:
        if self._server is not None:
            self._server.stop()
        self._scene = None
        self._server = None
        self._status_markdown = None
        self._plan_html = None
        self._graph_html = None
        self._events_html = None
        self._progress = None
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
                    "modes execute the verified RoboCasa trajectory."
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

            @run_goal.on_click
            def _(_) -> None:
                try:
                    self.controls.submit_goal(
                        goal.value,
                        planner.value,
                        execution_mode.value,
                    )
                except (RuntimeError, ValueError) as exc:
                    run_goal.hint = str(exc)
                self.refresh_controls()

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
            def _(event) -> None:
                preset = self._camera_presets[camera.value]
                clients = (
                    (event.client,)
                    if event.client is not None
                    else tuple(self._server.get_clients().values())
                )
                for client in clients:
                    _apply_camera_preset(client, preset)

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
            _apply_camera_preset(client, self._camera_presets["Agent"])


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
        else "verified"
    )
    return (
        '<div style="font-family: Inter, ui-sans-serif, system-ui, sans-serif; '
        'padding: 6px 2px 12px; color: #172033;">'
        '<div style="display:flex; align-items:center; justify-content:space-between; '
        'margin-bottom:12px;">'
        '<strong style="font-size:17px;">Live Retriever Flow</strong>'
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
    stage_rows: list[str] = []
    for stage_index, (_stage_id, stage_label, stage_steps) in enumerate(
        stages, start=1
    ):
        rows: list[str] = []
        states: list[str] = []
        for index, step in stage_steps:
            if step.step_id == snapshot.current_step_id:
                state, color, symbol = "Running", "#2563eb", "&#9654;"
            elif step.step_id in completed:
                state, color, symbol = "Completed", "#15803d", "&#10003;"
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
        if all(state == "Completed" for state in states):
            stage_state, stage_color, stage_symbol = "Done", "#15803d", "&#10003;"
        elif "Running" in states:
            stage_state, stage_color, stage_symbol = (
                "Running",
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

    up = (0.0, 0.0, 1.0)
    return {
        "Robot": _CameraPreset(
            position=_vector_sum(
                _scaled(forward, -1.0),
                _scaled(up, 2.6),
            ),
            look_at=_vector_sum(_scaled(forward, 0.3), _scaled(up, 1.0)),
            fov_degrees=65.0,
        ),
        "Agent": _CameraPreset(
            position=_vector_sum(_scaled(forward, -0.6), _scaled(up, 1.85)),
            look_at=_vector_sum(_scaled(forward, 1.0), _scaled(up, 0.8)),
            fov_degrees=60.0,
        ),
        "Overview": _CameraPreset(
            position=_vector_sum(
                _scaled(forward, -0.6),
                _scaled(up, 5.0),
            ),
            look_at=_vector_sum(_scaled(forward, 0.5), _scaled(up, 0.7)),
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
