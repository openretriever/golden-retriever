from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, order=True)
class GroundAtom:
    predicate: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.predicate
        return f"{self.predicate}({', '.join(self.args)})"


SymbolicState = frozenset[GroundAtom]


@dataclass(frozen=True, order=True)
class GroundAction:
    name: str
    args: tuple[str, ...] = ()

    def __str__(self) -> str:
        if not self.args:
            return self.name
        return f"{self.name}({', '.join(self.args)})"


@dataclass(frozen=True)
class GoalSpec:
    """Task-level goal description at the TAMP boundary.

    The initial scaffold keeps goals intentionally narrow: a conjunction of required
    atoms. A richer goal surface (preferences, costs, soft constraints) can grow
    later without changing the controller's high-level contract.
    """

    required_atoms: SymbolicState = frozenset()

    def is_satisfied_by(self, state: SymbolicState) -> bool:
        return set(self.required_atoms).issubset(state)


@dataclass(frozen=True)
class WorldSnapshot:
    """Concrete world state at the TAMP boundary.

    `raw_observation` stays intentionally untyped so GoldenRetriever bridges can
    inject simulator state, robot state, vision results, or flow payloads.
    """

    raw_observation: Any = None
    symbolic_state: SymbolicState = frozenset()
    objects: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
