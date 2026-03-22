from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from retriever_tamp.core.types import GoalSpec, GroundAction, GroundAtom, SymbolicState, WorldSnapshot


@dataclass(frozen=True)
class OperatorSchema:
    name: str
    parameters: tuple[tuple[str, str], ...] = ()
    preconditions: tuple[GroundAtom, ...] = ()
    add_effects: tuple[GroundAtom, ...] = ()
    delete_effects: tuple[GroundAtom, ...] = ()


@dataclass(frozen=True)
class TaskPlanningProblem:
    initial_state: SymbolicState
    goal: GoalSpec
    operators: tuple[OperatorSchema, ...]

    @property
    def goal_atoms(self) -> SymbolicState:
        """Compatibility convenience for planners that still think in raw atoms."""
        return self.goal.required_atoms


class SymbolicModel(Protocol):
    def abstract(self, snapshot: WorldSnapshot) -> SymbolicState:
        """Project a concrete snapshot into symbolic state."""

    def operators(self, snapshot: WorldSnapshot) -> Sequence[OperatorSchema]:
        """Return operator schemas relevant for this snapshot/problem."""

    def goal(self, snapshot: WorldSnapshot) -> GoalSpec:
        """Return the task-level goal for the active problem instance."""


class TaskPlanner(Protocol):
    def plan(self, problem: TaskPlanningProblem) -> Sequence[GroundAction]:
        """Return a symbolic plan, usually search over grounded actions."""
