from retriever.flow.clock import Hybrid
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

# Run: pixi run demo-manipulation
import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger, Flow, flow_io
from retriever.types.options import Action

from ..flows.env_manipulation import ManipulationEnvFlow
from ..flows.perception import PerceptionFlow
from ..flows.belief_updater import BeliefUpdaterFlow
from ..flows.planner_astar import TaskPlannerFlow
from ..flows.skill_executor import SkillExecutorFlow
from ..flows.monitor_execution import ExecutionMonitorFlow
from ..types.flow_types import PlannerInput, ExecutorOutput

# --- Custom Flows for Demo Structure ---

class CameraSourceFlow(ManipulationEnvFlow):
    """Effectively the same as Env for this demo, acts as Source."""
    pass

class SkillPolicyFlow(SkillExecutorFlow):
    """
    Skill Executor acting as a high-level policy. 
    Outputs actions to be consumed by the RobotController.
    """
    pass

@flow_io
@dataclass
class ActionReturn:
    action: Optional[Action] = None

class RobotControllerSink(Flow[ExecutorOutput, ActionReturn]):
    """
    Acts as the sink that interfaces with the physical (or simulated) robot driver.
    Receives 'status' and 'action' from Policy, and sends 'action' to the hardware/sim.
    """
    def __init__(self, name: str = "RobotController"):
        self.name = name
    
    def step(self, inp: ExecutorOutput) -> ActionReturn:
        # In a real system, this would call the robot SDK (e.g., Bosdyn API)
        # Here we just pass it through to the Mock Environment
        if inp.action:
            print(f"[{self.name}] Commanding Action: {inp.action}")
            return ActionReturn(action=inp.action)
        return ActionReturn(action=None)


def build_manipulation_pipeline():
    pipe = Pipeline("manipulation_pipeline")

    env = ManipulationEnvFlow(name="ManipulationEnv") @ Rate(10.0)
    perception = PerceptionFlow(name="Perception") @ Trigger("camera_images")
    belief = BeliefUpdaterFlow(name="BeliefState") @ Trigger("visible_atoms")
    planner = TaskPlannerFlow(name="TaskPlanner") @ Trigger("state")
    policy = SkillPolicyFlow(name="SkillPolicy") @ Trigger("plan")
    controller = RobotControllerSink(name="RobotController") @ Trigger("action")
    monitor = ExecutionMonitorFlow(name="Monitor") @ Hybrid(hz=10, trigger="status")

    # Env -> Perception
    pipe.connect(env, perception, map={"camera_images": "data"}, sync=Latest())
    
    # Perception -> Belief
    pipe.connect(perception, belief, map={"atoms": "visible_atoms"}, sync=Latest())
    
    # Simple Planner -> Belief (Simulate 'plan' memory)
    pipe.connect(planner, belief, map={"plan": "plan"}, sync=Latest())
    
    # Belief -> Planner (State Estimate)
    pipe.connect(belief, planner, map={"belief": "state"}, sync=Latest())
    
    # Planner -> Policy (Execute Plan)
    pipe.connect(planner, policy, map={"plan": "plan"}, sync=Latest())
    
    # Policy -> Monitor (Status updates)
    pipe.connect(policy, monitor, map={"status": "executor_status"}, sync=Latest())
    
    # Policy -> Belief (Action Feedback Loop)
    pipe.connect(policy, belief, map={"action": "action"}, sync=Latest())
    
    # Policy -> RobotController (Command Interface)
    pipe.connect(policy, controller, map={"action": "action", "status": "status"}, sync=Latest())
    
    # RobotController -> Env (Simulated Actuation Loop)
    pipe.connect(controller, env, map={"action": "action"}, sync=Latest())
    
    return pipe

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    # Build
    pipe = build_manipulation_pipeline()
    
    # Run
    # Configure Backend (Dora + Rerun)
    rerun_config = {"connect_addr": "127.0.0.1:9876"}
    
    pipe.run(
        duration=args.duration, 
        backend="dora",
        backend_config={
             "dora_timeout": 10,
             "env_overrides": {"GEMINI_API_KEY": "TODO"},
             "rerun_config": rerun_config
        }
    )
    
    # Generate Viz
    from retriever.ir.viz import save_interactive_html
    # pipe.build_ir() returns the IR
    save_interactive_html(pipe.build_ir(), "viz-manipulation-pipeline.html")
