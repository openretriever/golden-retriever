from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol

from retriever_tamp.core.types import WorldSnapshot


@dataclass(frozen=True)
class WorldDefinition:
    name: str
    constants: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ProblemDefinition:
    name: str
    world: WorldDefinition
    initial_snapshot: WorldSnapshot | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


class ProblemFactory(Protocol):
    def build(self, **kwargs: Any) -> ProblemDefinition:
        """Create a concrete problem instance for a domain/world."""
