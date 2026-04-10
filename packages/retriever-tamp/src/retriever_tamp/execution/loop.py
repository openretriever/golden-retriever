from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol, Sequence

from retriever_tamp.core.types import GroundAction, WorldSnapshot
from retriever_tamp.refinement.base import RefinementProvider, RefinementRequest, RefinementResult
from retriever_tamp.symbolic.base import SymbolicModel, TaskPlanner, TaskPlanningProblem


class ReplanReason(str, Enum):
    GOAL_REACHED = "goal_reached"
    NO_PLAN = "no_plan"
    REFINEMENT_FAILED = "refinement_failed"
    EXECUTION_FAILED = "execution_failed"
    MONITOR_TRIGGER = "monitor_trigger"
    STEP_EXECUTED = "step_executed"


@dataclass(frozen=True)
class ExecutionFeedback:
    success: bool
    completed_action: GroundAction | None = None
    message: str = ""
    payload: Mapping[str, Any] = field(default_factory=dict)


class ExecutionAdapter(Protocol):
    def execute(self, refinement: RefinementResult) -> ExecutionFeedback:
        """Execute one refined step in simulation or on a robot."""


class ExecutionMonitor(Protocol):
    def should_replan(
        self,
        *,
        snapshot: WorldSnapshot,
        current_plan: Sequence[GroundAction],
        last_feedback: ExecutionFeedback | None,
    ) -> bool:
        """Return whether the controller should discard the current plan."""


@dataclass
class TAMPController:
    symbolic_model: SymbolicModel
    task_planner: TaskPlanner
    refinement_provider: RefinementProvider
    execution_adapter: ExecutionAdapter
    execution_monitor: ExecutionMonitor | None = None

    def step(
        self,
        snapshot: WorldSnapshot,
    ) -> tuple[ReplanReason, Sequence[GroundAction], RefinementResult | None, ExecutionFeedback | None]:
        goal = self.symbolic_model.goal(snapshot)
        symbolic_state = frozenset(self.symbolic_model.abstract(snapshot))

        if goal.is_satisfied_by(symbolic_state):
            return ReplanReason.GOAL_REACHED, (), None, None

        problem = TaskPlanningProblem(
            initial_state=symbolic_state,
            goal=goal,
            operators=tuple(self.symbolic_model.operators(snapshot)),
        )
        plan = tuple(self.task_planner.plan(problem))
        if not plan:
            return ReplanReason.NO_PLAN, (), None, None

        next_action = plan[0]
        refinement = self.refinement_provider.refine(
            RefinementRequest(action=next_action, snapshot=snapshot)
        )
        if not refinement.success:
            return ReplanReason.REFINEMENT_FAILED, plan, refinement, None

        feedback = self.execution_adapter.execute(refinement)
        if not feedback.success:
            return ReplanReason.EXECUTION_FAILED, plan, refinement, feedback

        if self.execution_monitor is not None and self.execution_monitor.should_replan(
            snapshot=snapshot,
            current_plan=plan,
            last_feedback=feedback,
        ):
            return ReplanReason.MONITOR_TRIGGER, plan, refinement, feedback

        return ReplanReason.STEP_EXECUTED, plan, refinement, feedback
