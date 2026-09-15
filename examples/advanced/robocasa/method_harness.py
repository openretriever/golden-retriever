"""Safe, typed harness for embodied code-as-policy methods.

Methods produce the existing PolicyObservation to ActionChunk contract. The
harness validates each future-action chunk before an environment adapter can
execute it; it never evaluates generated Python or exposes simulator objects
to a planner.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

import numpy as np
from retriever.flow import io

from examples.advanced.openpi_policy.common import ActionChunk, PolicyObservation


@io
@dataclass(frozen=True)
class TrialRequest:
    """Reproducible configuration for one method trial."""

    method_id: str = ""
    task: str = ""
    episode: int = 0
    seed: int = 0
    max_steps: int = 500


@io
@dataclass(frozen=True)
class Transition:
    """Result of applying one validated low-level action."""

    observation: PolicyObservation | None = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False


@io
@dataclass(frozen=True)
class Verification:
    """Task-level result reported by the environment's native evaluator."""

    success: bool = False
    reward: float = 0.0
    message: str = ""


@io
@dataclass(frozen=True)
class HarnessEvent:
    """One ordered event in an inspectable trial lifecycle."""

    sequence: int = 0
    kind: str = "trial_started"
    status: str = "running"
    step: int = 0
    message: str = ""
    elapsed_seconds: float = 0.0


@io
@dataclass(frozen=True)
class TrialReport:
    """Compact, serializable result for reproduced method runs."""

    request: TrialRequest = TrialRequest()
    status: str = "pending"
    success: bool = False
    steps: int = 0
    total_reward: float = 0.0
    verification: Verification = Verification()
    events: tuple[HarnessEvent, ...] = ()
    elapsed_seconds: float = 0.0
    error: str = ""


class EnvironmentAdapter(Protocol):
    """Minimal simulator boundary required by the method harness."""

    def reset(self, request: TrialRequest) -> PolicyObservation: ...

    def step(self, action: np.ndarray) -> Transition: ...

    def verify(self) -> Verification: ...

    def close(self) -> None: ...


class MethodAdapter(Protocol):
    """A policy that plans a bounded chunk of future low-level actions."""

    def predict(self, observation: PolicyObservation) -> ActionChunk: ...


@dataclass(frozen=True)
class SafetyEnvelope:
    """Shape and numeric bounds enforced before simulator dispatch."""

    max_horizon: int = 64
    max_dof: int = 32
    max_abs_action: float = 1.0
    allowed_sources: frozenset[str] = frozenset({"scripted"})

    def validate(self, chunk: ActionChunk) -> ActionChunk:
        actions = np.asarray(chunk.actions, dtype=np.float32)
        if actions.ndim != 2:
            raise ValueError("ActionChunk.actions must have shape (horizon, dof)")
        if chunk.horizon != actions.shape[0] or chunk.dof != actions.shape[1]:
            raise ValueError("ActionChunk metadata does not match its array shape")
        if not 1 <= chunk.horizon <= self.max_horizon:
            raise ValueError("ActionChunk horizon exceeds the safety envelope")
        if not 1 <= chunk.dof <= self.max_dof:
            raise ValueError("ActionChunk dof exceeds the safety envelope")
        if not np.isfinite(actions).all():
            raise ValueError("ActionChunk contains a non-finite action")
        if np.abs(actions).max(initial=0.0) > self.max_abs_action:
            raise ValueError("ActionChunk exceeds the configured action bound")
        if chunk.source not in self.allowed_sources:
            raise ValueError(f"ActionChunk source is not allowed: {chunk.source}")
        return ActionChunk(
            actions=np.ascontiguousarray(actions),
            horizon=chunk.horizon,
            dof=chunk.dof,
            source=chunk.source,
        )


class MethodHarness:
    """Execute and report one bounded embodied-method trial."""

    def __init__(
        self,
        *,
        safety: SafetyEnvelope | None = None,
        event_sink: Callable[[HarnessEvent], None] | None = None,
    ) -> None:
        self.safety = safety or SafetyEnvelope()
        self.event_sink = event_sink

    def run(
        self,
        request: TrialRequest,
        *,
        environment: EnvironmentAdapter,
        method: MethodAdapter,
        close_environment: bool = True,
    ) -> TrialReport:
        started_at = monotonic()
        events: list[HarnessEvent] = []
        steps = 0
        total_reward = 0.0
        verification = Verification()
        error = ""

        def emit(kind: str, status: str, message: str = "") -> None:
            event = HarnessEvent(
                sequence=len(events),
                kind=kind,
                status=status,
                step=steps,
                message=message,
                elapsed_seconds=monotonic() - started_at,
            )
            events.append(event)
            if self.event_sink is not None:
                with suppress(Exception):
                    self.event_sink(event)

        emit("trial_started", "running", request.task)
        status = "error"
        try:
            _validate_request(request)
            observation = environment.reset(request)
            emit("environment_ready", "running")
            terminal = False
            while steps < request.max_steps and not terminal:
                chunk = self.safety.validate(method.predict(observation))
                emit(
                    "chunk_dispatched",
                    "running",
                    f"{chunk.horizon} future actions from {chunk.source}",
                )
                for action in chunk.actions:
                    transition = environment.step(action)
                    steps += 1
                    total_reward += float(transition.reward)
                    terminal = transition.terminated or transition.truncated
                    emit("step_completed", "running")
                    if transition.observation is not None:
                        observation = transition.observation
                    elif not terminal:
                        raise RuntimeError(
                            "Environment returned no observation before termination"
                        )
                    if terminal or steps >= request.max_steps:
                        break

            verification = environment.verify()
            status = "success" if verification.success else "failed"
            emit(
                "verification",
                status,
                verification.message or "Task verification complete",
            )
        except Exception as exc:  # noqa: BLE001 - trial boundary records failures
            error = f"{type(exc).__name__}: {exc}"
            emit("trial_error", "error", error)
            status = "error"
        finally:
            if close_environment:
                try:
                    environment.close()
                except Exception as exc:  # noqa: BLE001 - cleanup is reported
                    if not error:
                        error = f"{type(exc).__name__}: {exc}"
                        status = "error"
                        emit("trial_error", "error", error)

        emit("trial_completed", status)
        return TrialReport(
            request=request,
            status=status,
            success=status == "success",
            steps=steps,
            total_reward=total_reward,
            verification=verification,
            events=tuple(events),
            elapsed_seconds=monotonic() - started_at,
            error=error,
        )


def _validate_request(request: TrialRequest) -> None:
    if not request.method_id.strip():
        raise ValueError("TrialRequest.method_id is required")
    if not request.task.strip():
        raise ValueError("TrialRequest.task is required")
    if request.episode < 0:
        raise ValueError("TrialRequest.episode must be nonnegative")
    if request.seed < 0:
        raise ValueError("TrialRequest.seed must be nonnegative")
    if request.max_steps <= 0:
        raise ValueError("TrialRequest.max_steps must be positive")


__all__ = [
    "EnvironmentAdapter",
    "HarnessEvent",
    "MethodAdapter",
    "MethodHarness",
    "SafetyEnvelope",
    "Transition",
    "TrialReport",
    "TrialRequest",
    "Verification",
]
