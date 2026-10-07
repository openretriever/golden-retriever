"""The world boundary: reset, inspect, verify, close.

Deliberately shaped to mirror `method_harness.EnvironmentAdapter`
(`reset` / `step` / `verify` / `close`) so the two converge, with one
difference that matters: a search agent acts on *containers*, so the
per-decision call is `inspect(container_id)` rather than `step(action)` over
raw action rows.

That gap is the open question for the project lead. A scripted skill executor
is what expands "inspect `drawer_middle`" into the `ActionChunk`s the harness
validates; until Phase 3 builds it, `MockSearchWorld` answers the same
questions with no simulator at all, which is what keeps Phase 1 testable in CI.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, runtime_checkable

from retriever.flow import io

# retriever_typing lives under src/, which the demo tasks' PYTHONPATH="." does not
# cover. Other lanes repeat this shim per-module (robotics_typing_standard/*.py);
# this is the only module here that needs it.
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "src"))

from retriever_typing import ExecutionStatus  # noqa: E402

from .search_belief import InspectionResult
from .search_scenes import SearchScene

__all__ = [
    "SearchObservation",
    "SearchVerification",
    "SearchWorld",
    "MockSearchWorld",
    "search_status",
]

# Status vocabulary comes from `retriever_typing.ExecutionStatus` rather than a
# local enum: it is the repo's exported standard type for exactly this, carries
# `progress` and `error_message` already, and AGENTS.md forbids redefining a
# standard type locally because type identity is the contract.
IN_PROGRESS = "IN_PROGRESS"
SUCCESS = "SUCCESS"
FAILURE = "FAILURE"


def search_status(
    *,
    inspected: int,
    total: int,
    found: bool,
    error_message: str | None = None,
) -> ExecutionStatus:
    """Progress and outcome for a search, in the repo's standard shape.

    `progress` is the fraction of the container space resolved, so it rises as
    containers are ruled out and reads 1.0 the moment the search is decided —
    either by finding the target or by exhausting every container. It is derived
    from counts only, so it stays deterministic and is safe to put in a trace.
    """
    if total <= 0:
        raise ValueError("a search needs at least one container")
    if found:
        return ExecutionStatus(status=SUCCESS, metadata={}, progress=1.0)
    if error_message:
        return ExecutionStatus(
            status=FAILURE,
            metadata={},
            progress=min(1.0, inspected / total),
            error_message=error_message,
        )
    if inspected >= total:
        return ExecutionStatus(
            status=FAILURE,
            metadata={},
            progress=1.0,
            error_message="every container was inspected and the target was not found",
        )
    return ExecutionStatus(
        status=IN_PROGRESS, metadata={}, progress=inspected / total
    )


@io
@dataclass(frozen=True)
class SearchObservation:
    """What the robot can see right now."""

    scene_id: str = ""
    opened: tuple[str, ...] = ()
    holding_target: bool = False
    step: int = 0


@io
@dataclass(frozen=True)
class SearchVerification:
    """The world's own verdict, independent of what the agent believes."""

    success: bool = False
    target_container: str = ""
    message: str = ""


@runtime_checkable
class SearchWorld(Protocol):
    """Minimum surface D1 needs from any backend, mock or real."""

    def reset(self, scene: SearchScene, seed: int) -> SearchObservation: ...

    def inspect(self, container_id: str) -> InspectionResult: ...

    def observe(self) -> SearchObservation: ...

    def status(self) -> ExecutionStatus: ...

    def verify(self) -> SearchVerification: ...

    def close(self) -> None: ...


@dataclass
class MockSearchWorld:
    """A world with no physics: the target is wherever the scene's seed says.

    Runs anywhere, needs nothing installed, and is exactly reproducible — which
    is what lets the belief, the policies and the trace be fully tested before
    Phase 3 introduces a simulator.
    """

    scene: SearchScene | None = None
    seed: int = 0
    _target_container: str = ""
    _opened: list[str] = field(default_factory=list)
    _step: int = 0
    _found: bool = False

    def reset(self, scene: SearchScene, seed: int) -> SearchObservation:
        self.scene = scene
        self.seed = seed
        self._target_container = scene.target_container(seed)
        self._opened = []
        self._step = 0
        self._found = False
        return self.observe()

    def inspect(self, container_id: str) -> InspectionResult:
        if self.scene is None:
            raise RuntimeError("reset() the world before inspecting it")
        self.scene.container(container_id)  # raises for an unknown container
        self._step += 1
        if container_id not in self._opened:
            self._opened.append(container_id)
        found = container_id == self._target_container
        self._found = self._found or found
        return InspectionResult(container_id=container_id, found=found, step=self._step)

    def observe(self) -> SearchObservation:
        return SearchObservation(
            scene_id=self.scene.scene_id if self.scene else "",
            opened=tuple(self._opened),
            holding_target=self._found,
            step=self._step,
        )

    def status(self) -> ExecutionStatus:
        if self.scene is None:
            raise RuntimeError("reset() the world before asking for its status")
        return search_status(
            inspected=len(self._opened),
            total=len(self.scene.containers),
            found=self._found,
        )

    def verify(self) -> SearchVerification:
        return SearchVerification(
            success=self._found,
            target_container=self._target_container,
            message=(
                f"target was in {self._target_container}"
                if self._found
                else f"target was in {self._target_container} and was never found"
            ),
        )

    def close(self) -> None:  # nothing to release
        return None
