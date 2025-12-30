from dataclasses import dataclass, field
from typing import List, Optional, Set

from retriever.flow import Flow, flow_io
from retriever.types.options import Action, Option
from retriever.types.symbolic import GroundAtom, State

from .belief import BeliefState, BeliefUpdateInput, BeliefUpdateOutput
from .options import ReplanConfig


@flow_io
@dataclass
class EnvInput:
    action: Optional[Action] = None

@flow_io
@dataclass
class EnvOutput:
    data: dict

@flow_io
@dataclass
class PerceptionInput:
    data: dict

@flow_io
@dataclass
class PerceptionOutput:
    state: Optional[State] = None
    atoms: Set[GroundAtom] = field(default_factory=set)

@flow_io
@dataclass
class PlannerInput:
    """Input for heuristic planner."""
    state: Optional[BeliefState] = None
    replan_config: Optional[ReplanConfig] = None

@flow_io
@dataclass
class PlannerOutput:
    plan: List[Option]
    success: bool = True

    def __repr__(self):
        status = "Success" if self.success else "Failed"
        steps = [str(o) for o in self.plan]
        return f"PlannerOutput({status}, steps={len(self.plan)}): {steps}"

@flow_io
@dataclass
class ExecutorInput:
    state: Optional[BeliefState] = None
    plan: List[Option] = field(default_factory=list)

@flow_io
@dataclass
class ExecutorOutput:
    action: Optional[Action] = None
    status: str = "IDLE"
    failed_option: Optional[Option] = None

    def __repr__(self):
        act_str = f"Action({self.action.arr})" if self.action else "None"
        return f"ExecutorOutput(status={self.status}, action={act_str}, failed={self.failed_option})"

@flow_io
@dataclass
class MonitorInput:
    state: Optional[BeliefState] = None
    executor_status: Optional[str] = None

@flow_io
@dataclass
class MonitorOutput:
    replan_config: Optional[ReplanConfig] = None

    def __repr__(self):
        if self.replan_config and self.replan_config.should_replan:
             return f"MonitorOutput(REPLAN: {self.replan_config.reason})"
        return "MonitorOutput(OK)"

# --- Flow Type Aliases ---
EnvFlowType = Flow[EnvInput, EnvOutput]
PerceptionFlowType = Flow[PerceptionInput, PerceptionOutput]
BeliefUpdaterFlowType = Flow[BeliefUpdateInput, BeliefUpdateOutput]
PlannerFlowType = Flow[PlannerInput, PlannerOutput]
ExecutorFlowType = Flow[ExecutorInput, ExecutorOutput]
MonitorFlowType = Flow[MonitorInput, MonitorOutput]
