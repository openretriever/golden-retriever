from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set, Any

from retriever.flow import Flow, io
from retriever.types.options import Action, Option
from retriever.types.symbolic import GroundAtom, State

from .belief import BeliefState, BeliefUpdateInput, BeliefUpdateOutput
from .options import ReplanConfig


class ExecutionState(Enum):
    """State machine for execution monitor."""
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"


@io
@dataclass
class EnvInput:
    action: Optional[Action] = None

@io
@dataclass
class EnvOutput:
    data: dict

@io
@dataclass
class PerceptionInput:
    data: dict

@io
@dataclass
class PerceptionOutput:
    state: Optional[State] = None
    atoms: Set[GroundAtom] = field(default_factory=set)

@io
@dataclass
class PlannerInput:
    """Input for heuristic planner."""
    state: BeliefState = None  # type: ignore[assignment]
    replan_config: Optional[ReplanConfig] = None
    task: str = ""  # New task instruction field
    timestamp: float = 0.0  # For change detection

@io
@dataclass
class VLMPlannerInput:
    """Input for VLM task planner."""
    state: BeliefState = None  # type: ignore[assignment]
    task: str = ""
    timestamp: float = 0.0
    frame: Optional[bytes] = None

@io
@dataclass
class PlannerOutput:
    plan: List[Option]
    success: bool = True
    reasoning: str = ""
    belief_update: str = ""
    task: str = ""

    def __repr__(self):
        status = "Success" if self.success else "Failed"
        steps = [str(o) for o in self.plan]
        return f"PlannerOutput({status}, steps={len(self.plan)}): {steps}"

@io
@dataclass
class ExecutorInput:
    state: Optional[BeliefState] = None
    plan: List[Option] = field(default_factory=list)

@io
@dataclass
class ExecutorOutput:
    action: Optional[Action] = None
    status: str = "IDLE"
    plan: Optional[List[str]] = None  # Add plan support for VLM pipeline
    failed_option: Optional[Option] = None

    def __repr__(self):
        act_str = f"Action({self.action.arr})" if self.action else "None"
        return f"ExecutorOutput(status={self.status}, action={act_str}, failed={self.failed_option})"

@io
@dataclass
class MonitorInput:
    state: BeliefState = None  # type: ignore[assignment]
    observation: Optional[Any] = None
    executor_status: str = ""  # Changed from Optional
    
    # VLM Extensions
    frame: Optional[bytes] = None
    plan: List[Option] = field(default_factory=list)
    reasoning: Optional[str] = None
    belief_update: Optional[str] = None

@io
@dataclass
class MonitorOutput:
    replan_config: Optional[ReplanConfig] = None

    def __repr__(self):
        if self.replan_config and self.replan_config.should_replan:
             return f"MonitorOutput(REPLAN: {self.replan_config.reason})"
        return "MonitorOutput(OK)"
        
    # VLM Extensions
    state: Optional[ExecutionState] = None
    current_step: int = 0
    current_instruction: str = ""
    plan_display: str = ""
    task_completed: bool = False
    task_status: str = ""

# --- Flow Type Aliases ---
EnvFlowType = Flow[EnvInput, EnvOutput]
PerceptionFlowType = Flow[PerceptionInput, PerceptionOutput]
BeliefUpdaterFlowType = Flow[BeliefUpdateInput, BeliefUpdateOutput]
PlannerFlowType = Flow[PlannerInput, PlannerOutput]
ExecutorFlowType = Flow[ExecutorInput, ExecutorOutput]
MonitorFlowType = Flow[MonitorInput, MonitorOutput]
