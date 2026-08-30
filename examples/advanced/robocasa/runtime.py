"""Renderer-neutral runtime state for Retriever embodied demonstrations."""

from __future__ import annotations

from dataclasses import dataclass, replace
from threading import Event, RLock, Thread
from time import monotonic
from typing import Any

from .embodied import EmbodiedGoal, ExecutionEvent, SkillPlan


class PlannerUnavailableError(RuntimeError):
    """Raised when no planner is connected to the replay runtime."""


class PlannerBusyError(RuntimeError):
    """Raised when another planner request already owns the runtime."""


class PlannerTimeoutError(RuntimeError):
    """Raised when a planner exceeds its configured response deadline."""


class PlannerCancelledError(RuntimeError):
    """Raised when a planner response is invalidated before publication."""


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
    camera_preset: str = "Agent"
    planning: bool = False
    active_flow_node: str = "goal_source"
    plan: SkillPlan | None = None
    current_step_id: str = ""
    events: tuple[ExecutionEvent, ...] = ()


@dataclass(frozen=True)
class ReplayPresentation:
    """Canonical renderer-facing status derived from replay state."""

    status: str
    tone: str
    terminal: bool


def present_replay(snapshot: ReplaySnapshot) -> ReplayPresentation:
    """Project runtime state into a stable presentation contract."""

    status = "Planning" if snapshot.planning else snapshot.status
    if snapshot.success:
        status = "Success"
    tone = {
        "Success": "success",
        "Failed": "error",
        "Paused": "paused",
        "Verifying": "paused",
        "Planning": "running",
        "Running": "running",
        "Restarting": "running",
        "Stepping": "running",
        "Ready": "running",
    }.get(status, "idle")
    return ReplayPresentation(
        status=status,
        tone=tone,
        terminal=status in {"Success", "Failed"},
    )


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
        self._goal_accept_handler: Any | None = None
        self._goal_generation = 0
        self._active_planner_generation: int | None = None
        self._source_exhausted = False
        self._outcome_sealed = False

    def set_goal_handler(
        self,
        handler: Any,
        *,
        on_accept: Any | None = None,
    ) -> None:
        """Install the planner callback used by the browser goal composer."""

        with self._lock:
            self._goal_handler = handler
            self._goal_accept_handler = on_accept

    @property
    def can_submit_goals(self) -> bool:
        with self._lock:
            return self._goal_handler is not None

    def submit_goal(
        self,
        text: str,
        planner: str | None = None,
        execution_mode: str | None = None,
        *,
        timeout: float | None = None,
    ) -> SkillPlan:
        with self._lock:
            handler = self._goal_handler
            accept_handler = self._goal_accept_handler
            snapshot = self._snapshot
            if handler is None:
                raise PlannerUnavailableError("No embodied planner is connected")
            if self._active_planner_generation is not None:
                raise PlannerBusyError("A planner request is already running")
            self._goal_generation += 1
            generation = self._goal_generation
            self._active_planner_generation = generation
            self._snapshot = replace(
                self._snapshot,
                planning=True,
                active_flow_node="embodied_planner",
            )
        goal = EmbodiedGoal(
            text=text.strip(),
            task=snapshot.task,
            episode=snapshot.episode,
            planner=planner or snapshot.planner,
            execution_mode=execution_mode or snapshot.execution_mode,
        )
        timed_out = False
        try:
            plan = self._run_planner(
                handler,
                goal,
                timeout=timeout,
                generation=generation,
            )
            plan.validate()
            with self._lock:
                if generation != self._goal_generation:
                    raise PlannerCancelledError("Goal was cancelled")
                self._configure_execution_locked(goal, plan, emit_events=False)
                self._request_restart_locked(
                    event_message=(
                        f"Plan accepted from {plan.source} planner "
                        f"({goal.execution_mode.replace('_', ' ')})"
                    )
                )
                if accept_handler is not None:
                    accept_handler(plan.goal, plan)
            return plan
        except PlannerTimeoutError:
            timed_out = True
            raise
        finally:
            if not timed_out:
                self._finish_planner(generation)

    def _run_planner(
        self,
        handler: Any,
        goal: EmbodiedGoal,
        *,
        timeout: float | None,
        generation: int,
    ) -> SkillPlan:
        if timeout is None:
            return handler(goal)
        if timeout <= 0:
            raise ValueError("Planner timeout must be positive")

        completed = Event()
        release_when_complete = Event()
        result: dict[str, Any] = {}

        def plan_goal() -> None:
            try:
                result["plan"] = handler(goal)
            except Exception as exc:  # noqa: BLE001  # pragma: no cover
                result["error"] = exc
            finally:
                completed.set()
                if release_when_complete.is_set():
                    self._finish_planner(generation)

        Thread(
            target=plan_goal,
            name="retriever-embodied-planner",
            daemon=True,
        ).start()
        if not completed.wait(timeout=timeout):
            release_when_complete.set()
            if completed.is_set():
                self._finish_planner(generation)
            raise PlannerTimeoutError(f"Planner timed out after {timeout:g} seconds")
        error = result.get("error")
        if error is not None:
            raise error
        plan = result.get("plan")
        if not isinstance(plan, SkillPlan):
            raise TypeError("Planner must return a SkillPlan")
        return plan

    def _finish_planner(self, generation: int) -> None:
        with self._lock:
            if self._active_planner_generation != generation:
                return
            self._active_planner_generation = None
            self._snapshot = replace(
                self._snapshot,
                planning=False,
                active_flow_node=(
                    "demo_actions" if self._snapshot.plan is not None else "goal_source"
                ),
            )

    def configure_execution(self, goal: EmbodiedGoal, plan: SkillPlan) -> None:
        plan.validate()
        with self._lock:
            self._goal_generation += 1
            self._configure_execution_locked(goal, plan)

    def cancel_pending_goals(self) -> None:
        """Prevent an in-flight planner response from mutating this run."""

        with self._lock:
            self._goal_generation += 1
            self._active_planner_generation = None
            self._snapshot = replace(
                self._snapshot,
                planning=False,
                active_flow_node=(
                    "demo_actions" if self._snapshot.plan is not None else "goal_source"
                ),
            )

    def snapshot(self) -> ReplaySnapshot:
        with self._lock:
            return self._snapshot

    def set_total_steps(self, total_steps: int) -> None:
        with self._lock:
            total_steps = max(0, int(total_steps))
            if total_steps == self._snapshot.total_steps:
                return
            if self._restart_requested:
                self._snapshot = replace(self._snapshot, total_steps=total_steps)
                return
            self._reset_lifecycle_locked(
                cycle=self._snapshot.cycle,
                status="Ready",
                event_message="Replay initialized",
                total_steps=total_steps,
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

    def begin_repeat_cycle(self) -> int:
        """Advance an automatic repeat without desynchronizing UI state."""

        with self._lock:
            previous = (self._snapshot.cycle, self._snapshot.success)
            self._begin_cycle_locked(
                self._snapshot.cycle + 1,
                "Running",
                previous_outcome=previous,
            )
            return self._snapshot.cycle

    def set_speed(self, speed: float) -> None:
        if speed <= 0:
            raise ValueError("Replay speed must be positive")
        with self._lock:
            self._snapshot = replace(self._snapshot, speed=float(speed))

    def set_camera_preset(self, preset: str) -> None:
        if preset not in {"Robot", "Agent", "Overview"}:
            raise ValueError(f"Unknown camera preset: {preset}")
        with self._lock:
            self._snapshot = replace(self._snapshot, camera_preset=preset)

    def claim_next_action(self) -> tuple[bool, int | None]:
        """Return whether the source may advance and an atomic restart cycle."""

        with self._lock:
            if self._restart_requested:
                self._restart_requested = False
                return True, self._snapshot.cycle
            if self._is_terminal_locked():
                return False, None
            if not self._snapshot.paused:
                return True, None
            if self._step_budget > 0:
                self._step_budget -= 1
                return True, None
            return False, None

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
            if int(cycle) != self._snapshot.cycle:
                return
            if int(episode_step) < self._snapshot.episode_step:
                return
            if self._outcome_sealed:
                return
            final_observation = (
                self._snapshot.total_steps > 0
                and int(episode_step) >= self._snapshot.total_steps - 1
            )
            verified = bool(success) and final_observation
            if self._source_exhausted and final_observation:
                self._outcome_sealed = True
            if verified:
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
                success=verified,
                action_norm=float(action_norm),
                active_flow_node=(
                    "event_sink"
                    if verified
                    else "task_verifier"
                    if self._source_exhausted
                    else "robocasa_simulator"
                ),
            )
            self._update_plan_events(
                self._snapshot.progress,
                verified,
                terminal_failure=self._source_exhausted and final_observation,
            )

    def mark_complete(self) -> None:
        with self._lock:
            self._source_exhausted = True
            final_observation = (
                self._snapshot.total_steps > 0
                and self._snapshot.episode_step >= self._snapshot.total_steps - 1
            )
            verified = self._snapshot.success and final_observation
            if final_observation:
                self._outcome_sealed = True
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
                active_flow_node=(
                    "event_sink" if verified or final_observation else "task_verifier"
                ),
            )
            self._update_plan_events(
                self._snapshot.progress,
                verified,
                terminal_failure=final_observation and not verified,
            )

    def _is_terminal_locked(self) -> bool:
        # A success observation still needs one source call so the source can
        # either mark exhaustion or explicitly begin an automatic repeat.
        return self._source_exhausted

    def _configure_execution_locked(
        self,
        goal: EmbodiedGoal,
        plan: SkillPlan,
        *,
        emit_events: bool = True,
    ) -> None:
        self._started_at = monotonic()
        self._event_sequence = 0
        self._source_exhausted = False
        self._outcome_sealed = False
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
            paused=False,
            episode_step=0,
            progress=0.0,
            reward=0.0,
            success=False,
            action_norm=0.0,
            status="Ready",
            planning=False,
            active_flow_node="demo_actions",
        )
        if emit_events:
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

    def _request_restart_locked(
        self,
        *,
        event_message: str = "Episode restarted",
    ) -> None:
        if not self._restart_requested:
            next_cycle = self._snapshot.cycle + 1
        else:
            next_cycle = self._snapshot.cycle
        self._restart_requested = True
        self._begin_cycle_locked(
            next_cycle,
            "Restarting",
            event_message=event_message,
        )

    def _begin_cycle_locked(
        self,
        cycle: int,
        status: str,
        *,
        event_message: str = "Episode restarted",
        previous_outcome: tuple[int, bool] | None = None,
    ) -> None:
        self._reset_lifecycle_locked(
            cycle=cycle,
            status=status,
            event_message=event_message,
            previous_outcome=previous_outcome,
        )

    def _reset_lifecycle_locked(
        self,
        *,
        cycle: int,
        status: str,
        event_message: str,
        total_steps: int | None = None,
        previous_outcome: tuple[int, bool] | None = None,
    ) -> None:
        self._source_exhausted = False
        self._outcome_sealed = False
        self._step_budget = 0
        plan = self._snapshot.plan
        self._started_at = monotonic()
        self._event_sequence = 0
        self._snapshot = replace(
            self._snapshot,
            paused=False,
            episode_step=0,
            cycle=cycle,
            progress=0.0,
            reward=0.0,
            success=False,
            action_norm=0.0,
            status=status,
            active_flow_node="demo_actions",
            total_steps=(
                self._snapshot.total_steps if total_steps is None else total_steps
            ),
            current_step_id=plan.steps[0].step_id if plan is not None else "",
            events=(),
        )
        if previous_outcome is not None:
            previous_cycle, succeeded = previous_outcome
            self._append_event(
                kind="cycle",
                status="verified" if succeeded else "failed",
                step_id="",
                message=(
                    f"Cycle {previous_cycle} verified"
                    if succeeded
                    else f"Cycle {previous_cycle} completed without verification"
                ),
            )
        if plan is not None:
            self._append_event(
                kind="dispatch",
                status="completed",
                step_id="",
                message=event_message,
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
                    event for event in self._snapshot.events if event.status != "failed"
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
            verification = next(
                (
                    step
                    for step in plan.steps
                    if step.step_id == plan.verification_step_id
                ),
                None,
            )
            if (
                verification is not None
                and (verification.step_id, "failed") not in existing
            ):
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
            events=(*self._snapshot.events, event),
        )
