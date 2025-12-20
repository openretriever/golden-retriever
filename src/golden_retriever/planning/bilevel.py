"""
Bilevel Closed-Loop Planning Architecture

Implements the core bilevel planning system from Proposal v2.7:
- Strategic Layer (1Hz): High-level task decomposition and skill selection
- Tactical Layer (30Hz): Real-time skill policy execution with reactive control
- FRP Coordination: Automatic temporal coordination between layers
- Closed-Loop Monitoring: Continuous feedback and adaptive replanning

This module provides the main BilevelPlanningSystem that coordinates strategic 
and tactical planning with automatic FRP temporal coordination.
"""

from typing import Generic, TypeVar, Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import time
from enum import Enum

from ..core.types import Flow, Pipeline, Eff, RGBImage
from ..core.frp_engine import flow
from ..core.temporal import ExecutionTimer

# Type variables for generic planning
I = TypeVar('I')
O = TypeVar('O')
S = TypeVar('S')  # State type


# ======================== CORE TYPES ========================

@dataclass(frozen=True)
class TaskRequest:
    """High-level task request from user."""
    description: str
    goal_state: Optional[Dict[str, Any]] = None
    constraints: List[str] = field(default_factory=list)
    priority: str = "normal"
    timeout: Optional[float] = None


@dataclass(frozen=True) 
class StrategicPlan:
    """High-level strategic plan with skill sequence."""
    task_id: str
    skills: List['SkillInstruction']
    goal_state: Dict[str, Any]
    estimated_duration: float
    constraints: List[str] = field(default_factory=list)
    
    def remaining_skills(self, current_index: int) -> List['SkillInstruction']:
        """Get remaining skills from current execution point."""
        return self.skills[current_index:]


@dataclass(frozen=True)
class SkillInstruction:
    """Individual skill instruction in strategic plan."""
    skill_name: str
    parameters: Dict[str, Any]
    preconditions: List[str] = field(default_factory=list)
    postconditions: List[str] = field(default_factory=list)
    timeout: float = 30.0


@dataclass(frozen=True)
class TacticalPlan:
    """Low-level tactical plan for skill execution."""
    skill_instruction: SkillInstruction
    action_sequence: List['RobotAction']
    control_parameters: Dict[str, Any] = field(default_factory=dict)
    safety_constraints: List[str] = field(default_factory=list)


@dataclass
class RobotAction:
    """Individual robot action in tactical execution."""
    action_type: str  # "move", "grasp", "release", etc.
    parameters: Dict[str, Any]
    expected_duration: float = 1.0
    safety_constraints: List[str] = field(default_factory=list)


@dataclass
class RobotState:
    """Current robot state for planning and execution."""
    pose: Optional[Dict[str, float]] = None
    joint_positions: Optional[List[float]] = None
    gripper_state: str = "open"
    held_objects: List[str] = field(default_factory=list)
    environment_objects: List[Dict[str, Any]] = field(default_factory=list)
    last_action_result: Optional[Dict[str, Any]] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)
    current_skill: Optional[str] = None
    
    def update_from_action(self, action: RobotAction, result: Dict[str, Any]) -> 'RobotState':
        """Create new state after action execution."""
        new_history = self.execution_history + [{"action": action, "result": result, "timestamp": time.time()}]
        return RobotState(
            pose=result.get("new_pose", self.pose),
            joint_positions=result.get("new_joint_positions", self.joint_positions),
            gripper_state=result.get("new_gripper_state", self.gripper_state),
            held_objects=result.get("new_held_objects", self.held_objects),
            environment_objects=result.get("new_environment_objects", self.environment_objects),
            last_action_result=result,
            execution_history=new_history,
            current_skill=self.current_skill
        )


class PlanningResult(Enum):
    """Results of planning operations."""
    SUCCESS = "success"
    FAILURE = "failure" 
    REPLANNING_NEEDED = "replanning_needed"
    TIMEOUT = "timeout"


@dataclass
class ExecutionResult:
    """Result of tactical execution."""
    result: PlanningResult
    final_state: RobotState
    actions_executed: List[RobotAction] = field(default_factory=list)
    execution_time: float = 0.0
    error_message: Optional[str] = None
    replanning_triggered: bool = False


# ======================== PLANNING INTERFACES ========================

class StrategicPlanner(Flow[Tuple[TaskRequest, RobotState], StrategicPlan]):
    """Strategic planner interface for high-level task decomposition."""
    
    @abstractmethod
    def run(self, inputs: Tuple[TaskRequest, RobotState]) -> StrategicPlan:
        """Create strategic plan from task request and current state."""
        pass


class TacticalPlanner(Flow[Tuple[SkillInstruction, RobotState], TacticalPlan]):
    """Tactical planner interface for skill-level execution planning."""
    
    @abstractmethod  
    def run(self, inputs: Tuple[SkillInstruction, RobotState]) -> TacticalPlan:
        """Create tactical plan for specific skill execution."""
        pass


class SkillPolicy(Flow[Tuple[RobotState, TacticalPlan], Eff[RobotState, RobotAction]]):
    """Skill policy interface for real-time action generation."""
    
    @abstractmethod
    def run(self, inputs: Tuple[RobotState, TacticalPlan]) -> Eff[RobotState, RobotAction]:
        """Generate next action based on current state and tactical plan."""
        pass


# ======================== MONITORING INTERFACES ========================

class PlanningMonitor(ABC):
    """Interface for monitoring planning and execution."""
    
    @abstractmethod
    def should_replan_strategic(self, state: RobotState, execution_result: ExecutionResult) -> bool:
        """Determine if strategic replanning is needed."""
        pass
    
    @abstractmethod
    def should_replan_tactical(self, state: RobotState, action_result: Dict[str, Any]) -> bool:
        """Determine if tactical replanning is needed."""
        pass


# ======================== BILEVEL SYSTEM IMPLEMENTATION ========================

class BilevelPlanningSystem:
    """
    Main bilevel planning system implementing the architecture from Proposal v2.7.
    
    Coordinates strategic planning (1Hz) with tactical execution (30Hz) using
    FRP temporal coordination for automatic rate management and closed-loop control.
    """
    
    def __init__(
        self,
        strategic_planner: StrategicPlanner,
        tactical_planner: TacticalPlanner, 
        skill_policy: SkillPolicy,
        monitor: PlanningMonitor,
        strategic_rate: float = 1.0,  # 1Hz strategic planning
        tactical_rate: float = 30.0   # 30Hz tactical execution
    ):
        self.strategic_planner = strategic_planner
        self.tactical_planner = tactical_planner
        self.skill_policy = skill_policy
        self.monitor = monitor
        self.strategic_rate = strategic_rate
        self.tactical_rate = tactical_rate
        
        # Internal state for closed-loop operation
        self.current_strategic_plan: Optional[StrategicPlan] = None
        self.current_tactical_plan: Optional[TacticalPlan] = None
        self.strategic_step_index: int = 0
        
    def create_closed_loop_flow(self) -> 'ClosedLoopPlanningFlow':
        """Create the main closed-loop planning flow with FRP coordination."""
        return ClosedLoopPlanningFlow(self)


@flow(rate="1hz")  # Strategic layer runs at 1Hz
class StrategicPlanningFlow(Flow[Tuple[TaskRequest, RobotState], StrategicPlan]):
    """Strategic planning flow with automatic 1Hz rate coordination."""
    
    def __init__(self, planner: StrategicPlanner):
        self.planner = planner
    
    def run(self, inputs: Tuple[TaskRequest, RobotState]) -> StrategicPlan:
        return self.planner.run(inputs)


@flow(rate="10hz")  # Tactical planning at 10Hz (faster than strategic, slower than execution)
class TacticalPlanningFlow(Flow[Tuple[SkillInstruction, RobotState], TacticalPlan]):
    """Tactical planning flow with automatic 10Hz rate coordination."""
    
    def __init__(self, planner: TacticalPlanner):
        self.planner = planner
    
    def run(self, inputs: Tuple[SkillInstruction, RobotState]) -> TacticalPlan:
        return self.planner.run(inputs)


@flow(rate="30hz")  # Real-time execution at 30Hz
class SkillExecutionFlow(Flow[Tuple[RobotState, TacticalPlan], Eff[RobotState, RobotAction]]):
    """Skill execution flow with automatic 30Hz rate coordination."""
    
    def __init__(self, policy: SkillPolicy):
        self.policy = policy
    
    def run(self, inputs: Tuple[RobotState, TacticalPlan]) -> Eff[RobotState, RobotAction]:
        return self.policy.run(inputs)


class ClosedLoopPlanningFlow(Flow[TaskRequest, Eff[RobotState, ExecutionResult]]):
    """
    Main closed-loop planning flow that coordinates all bilevel planning.
    
    Implements the complete bilevel architecture:
    1. Strategic planning (1Hz) - Task decomposition and skill selection
    2. Tactical planning (10Hz) - Skill-specific action planning  
    3. Skill execution (30Hz) - Real-time reactive control
    4. Monitoring & Replanning - Continuous feedback and adaptation
    
    FRP coordination handles all temporal aspects automatically.
    """
    
    def __init__(self, bilevel_system: BilevelPlanningSystem):
        self.system = bilevel_system
        
        # Create FRP-coordinated flows
        self.strategic_flow = StrategicPlanningFlow(bilevel_system.strategic_planner)
        self.tactical_flow = TacticalPlanningFlow(bilevel_system.tactical_planner)
        self.execution_flow = SkillExecutionFlow(bilevel_system.skill_policy)
    
    def run(self, task_request: TaskRequest) -> Eff[RobotState, ExecutionResult]:
        """
        Execute complete bilevel planning with closed-loop control.
        
        This creates a description of the entire bilevel execution process,
        including strategic planning, tactical planning, execution, monitoring,
        and adaptive replanning - all with automatic FRP temporal coordination.
        """
        
        def bilevel_execution(initial_state: RobotState) -> Tuple[ExecutionResult, RobotState]:
            current_state = initial_state
            executed_actions = []
            start_time = time.time()
            
            try:
                # 1. STRATEGIC PLANNING (1Hz)
                # FRP automatically coordinates this at 1Hz rate
                strategic_plan = self.strategic_flow.run((task_request, current_state))
                self.system.current_strategic_plan = strategic_plan
                self.system.strategic_step_index = 0
                
                print(f"🎯 Strategic Plan: {len(strategic_plan.skills)} skills planned")
                
                # 2. EXECUTE STRATEGIC PLAN WITH CLOSED-LOOP CONTROL
                while self.system.strategic_step_index < len(strategic_plan.skills):
                    current_skill = strategic_plan.skills[self.system.strategic_step_index]
                    
                    # 3. TACTICAL PLANNING (10Hz)
                    # FRP automatically coordinates this at 10Hz rate
                    tactical_plan = self.tactical_flow.run((current_skill, current_state))
                    self.system.current_tactical_plan = tactical_plan
                    
                    print(f"⚡ Tactical Plan: {len(tactical_plan.action_sequence)} actions for {current_skill.skill_name}")
                    
                    # 4. SKILL EXECUTION WITH REAL-TIME CONTROL (30Hz)
                    skill_result, current_state = self._execute_skill_with_monitoring(
                        tactical_plan, current_state, executed_actions
                    )
                    
                    # 5. STRATEGIC MONITORING & REPLANNING
                    if self.system.monitor.should_replan_strategic(current_state, skill_result):
                        print("🔄 Strategic replanning triggered")
                        # Replan from current state
                        strategic_plan = self.strategic_flow.run((task_request, current_state))
                        self.system.current_strategic_plan = strategic_plan
                        self.system.strategic_step_index = 0
                        continue
                    
                    if skill_result.result != PlanningResult.SUCCESS:
                        return skill_result, current_state
                    
                    # Move to next strategic step
                    self.system.strategic_step_index += 1
                
                # All strategic steps completed successfully
                execution_time = time.time() - start_time
                return ExecutionResult(
                    result=PlanningResult.SUCCESS,
                    final_state=current_state,
                    actions_executed=executed_actions,
                    execution_time=execution_time
                ), current_state
                
            except Exception as e:
                execution_time = time.time() - start_time
                return ExecutionResult(
                    result=PlanningResult.FAILURE,
                    final_state=current_state,
                    actions_executed=executed_actions,
                    execution_time=execution_time,
                    error_message=str(e)
                ), current_state
        
        return Eff(bilevel_execution)
    
    def _execute_skill_with_monitoring(
        self,
        tactical_plan: TacticalPlan,
        current_state: RobotState,
        executed_actions: List[RobotAction]
    ) -> Tuple[ExecutionResult, RobotState]:
        """
        Execute tactical plan with real-time monitoring and adaptive control.
        
        This implements the 30Hz execution loop with continuous monitoring
        and tactical replanning as needed.
        """
        
        action_index = 0
        skill_start_time = time.time()
        
        while action_index < len(tactical_plan.action_sequence):
            # REAL-TIME ACTION EXECUTION (30Hz)
            # FRP automatically coordinates this at 30Hz rate
            action_eff = self.execution_flow.run((current_state, tactical_plan))
            
            # Execute action and update state
            action, new_state = action_eff.run(current_state)
            executed_actions.append(action)
            
            # Simulate action execution result
            action_result = self._simulate_action_execution(action, new_state)
            current_state = new_state.update_from_action(action, action_result)
            
            # TACTICAL MONITORING & REPLANNING
            if self.system.monitor.should_replan_tactical(current_state, action_result):
                print("⚡ Tactical replanning triggered")
                # Generate new tactical plan from current state
                tactical_plan = self.tactical_flow.run((tactical_plan.skill_instruction, current_state))
                action_index = 0  # Restart with new plan
                continue
            
            if not action_result.get("success", True):
                execution_time = time.time() - skill_start_time
                return ExecutionResult(
                    result=PlanningResult.FAILURE,
                    final_state=current_state,
                    actions_executed=executed_actions,
                    execution_time=execution_time,
                    error_message=f"Action {action.action_type} failed",
                    replanning_triggered=True
                ), current_state
            
            action_index += 1
        
        # Skill completed successfully
        execution_time = time.time() - skill_start_time
        return ExecutionResult(
            result=PlanningResult.SUCCESS,
            final_state=current_state,
            actions_executed=executed_actions,
            execution_time=execution_time
        ), current_state
    
    def _simulate_action_execution(self, action: RobotAction, state: RobotState) -> Dict[str, Any]:
        """Simulate action execution for demonstration."""
        # In real implementation, this would interface with robot hardware
        return {
            "success": True,
            "new_pose": state.pose,
            "execution_time": action.expected_duration,
            "timestamp": time.time()
        }


# ======================== FACTORY FUNCTIONS ========================

def create_bilevel_system(
    strategic_planner: StrategicPlanner,
    tactical_planner: TacticalPlanner,
    skill_policy: SkillPolicy,
    monitor: Optional[PlanningMonitor] = None
) -> BilevelPlanningSystem:
    """
    Factory function to create a complete bilevel planning system.
    
    This is the main entry point for creating bilevel planning systems
    with automatic FRP temporal coordination.
    """
    if monitor is None:
        monitor = create_default_monitor()
    
    return BilevelPlanningSystem(
        strategic_planner=strategic_planner,
        tactical_planner=tactical_planner,
        skill_policy=skill_policy,
        monitor=monitor
    )


def create_default_monitor() -> PlanningMonitor:
    """Create default monitoring implementation."""
    
    class DefaultPlanningMonitor(PlanningMonitor):
        def should_replan_strategic(self, state: RobotState, execution_result: ExecutionResult) -> bool:
            # Replan if execution failed or took too long
            return (execution_result.result == PlanningResult.FAILURE or 
                   execution_result.execution_time > 60.0)
        
        def should_replan_tactical(self, state: RobotState, action_result: Dict[str, Any]) -> bool:
            # Replan if action failed
            return not action_result.get("success", True)
    
    return DefaultPlanningMonitor()