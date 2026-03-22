from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence

from retriever_tamp.core.types import GroundAction, WorldSnapshot


@dataclass(frozen=True)
class RefinementRequest:
    action: GroundAction
    snapshot: WorldSnapshot
    context: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ExecutionPrimitive:
    name: str
    parameters: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementCandidate:
    label: str
    primitives: tuple[ExecutionPrimitive, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RefinementResult:
    action: GroundAction
    success: bool
    tried_candidates: tuple[str, ...] = ()
    candidate: RefinementCandidate | None = None
    failure_reason: str = ""


class RefinementProvider(Protocol):
    def refine(self, request: RefinementRequest) -> RefinementResult:
        """Produce an executable candidate for one symbolic step."""
