# Run: pixi run demo-manipulation
#
# Manipulation Pipeline (Real-World Interface)
#
# This pipeline demonstrates a manipulation robot architecture where:
# - Camera captures images at 30Hz (physical sensor rate)
# - High-level planning runs on belief state updates
# - Robot controller sends commands at 200Hz (servo rate)
#
# Unlike simulation pipelines, there is no "Env" flow - the physical world
# is the environment. Camera reads from hardware, controller sends to hardware.

import argparse
from dataclasses import dataclass
from typing import Optional

import retriever
from retriever.flow import Flow, Latest, Pipeline, Rate, Trigger, flow_io
from retriever.types.options import Action

from ..flows.belief_updater import BeliefUpdaterFlow
from ..flows.monitor_execution import ExecutionMonitorFlow
from ..flows.perception import PerceptionFlow
from ..flows.planner_astar import TaskPlannerFlow
from ..flows.skill_executor import SkillExecutorFlow
from ..types.flow_types import ExecutorOutput

# --- Source/Sink Flows for Physical Robot Interface ---

@flow_io
@dataclass
class CameraOutput:
    """Output from camera sensor."""
    data: dict  # Raw camera data (images, depth, etc.)


class CameraSourceFlow(Flow[None, CameraOutput]):
    """
    Camera source that reads from physical camera hardware.
    
    In a real deployment, this would interface with:
    - RealSense SDK
    - ZED SDK
    - ROS image topics
    
    For demo purposes, produces mock camera data.
    """
    def __init__(self, name: str = "CameraSource"):
        self.name = name
        self._frame_count = 0

    def step(self, inp: None) -> CameraOutput:
        self._frame_count += 1
        # Mock camera data - in reality, grab from hardware
        return CameraOutput(data={
            "rgb": f"frame_{self._frame_count}",
            "depth": None,
            "timestamp": self._frame_count / 30.0
        })


@flow_io
@dataclass
class ControlCommand:
    """Command sent to robot actuators."""
    action: Optional[Action] = None
    timestamp: float = 0.0


class RobotControllerSink(Flow[ExecutorOutput, ControlCommand]):
    """
    Robot controller sink that sends commands to hardware.
    
    In a real deployment, this would interface with:
    - Robot SDK (Bosdyn, Franka, UR, etc.)
    - ROS action servers
    - Low-level motor controllers
    
    Runs at 200Hz for smooth servo control.
    """
    def __init__(self, name: str = "RobotController"):
        self.name = name
        self._cmd_count = 0

    def step(self, inp: ExecutorOutput) -> ControlCommand:
        self._cmd_count += 1
        if inp.action:
            # In reality, send to robot hardware
            return ControlCommand(
                action=inp.action,
                timestamp=self._cmd_count / 200.0
            )
        return ControlCommand(action=None)


class SkillPolicyFlow(SkillExecutorFlow):
    """
    Skill executor acting as high-level policy.
    Translates plans into action commands for the controller.
    """
    pass


def build_manipulation_pipeline() -> Pipeline:
    """Build the manipulation pipeline for real-world robot control."""
    pipe = Pipeline("manipulation_pipeline")

    # --- Flow Initialization with Clock Policies ---

    # Camera captures at 30Hz (typical sensor rate)
    camera = CameraSourceFlow(name="CameraSource") @ Rate(30.0)

    # Perception triggers on new camera data
    perception = PerceptionFlow(name="PerceptionFlow") @ Trigger("data")

    # Belief updates on new observations
    belief = BeliefUpdaterFlow(name="BeliefUpdaterFlow") @ Trigger("observation")

    # Planner runs when belief state changes
    planner = TaskPlannerFlow(name="TaskPlannerFlow") @ Trigger("state")

    # Policy executes when plan or state updates
    policy = SkillPolicyFlow(name="SkillPolicyFlow") @ Trigger("state")

    # Controller runs at 200Hz for smooth control
    controller = RobotControllerSink(name="RobotControllerSink") @ Rate(200.0)

    # Monitor tracks execution status
    monitor = ExecutionMonitorFlow(name="PlanExecutionMonitorFlow") @ Trigger("executor_status")

    # --- Connections (matching rise pipeline structure) ---

    # Camera -> Perception
    pipe.connect(camera, perception, map={"data": "data"}, sync=Latest())

    # Perception -> Belief (state + atoms)
    pipe.connect(perception, belief, map={"state": "observation", "atoms": "visible_atoms"}, sync=Latest())

    # Policy -> Belief (action feedback)
    pipe.connect(policy, belief, map={"action": "action"}, sync=Latest())

    # Belief -> Planner
    pipe.connect(belief, planner, map={"belief": "state"}, sync=Latest())

    # Planner -> Policy
    pipe.connect(planner, policy, map={"plan": "plan"}, sync=Latest())

    # Planner -> Belief (plan memory)
    pipe.connect(planner, belief, map={"plan": "plan"}, sync=Latest())

    # Belief -> Policy (grounding context)
    pipe.connect(belief, policy, map={"belief": "state"}, sync=Latest())

    # Policy -> Controller
    pipe.connect(policy, controller, map={"action": "action", "status": "status"}, sync=Latest())

    # Policy -> Monitor
    pipe.connect(policy, monitor, map={"status": "executor_status"}, sync=Latest())

    # Belief -> Monitor
    pipe.connect(belief, monitor, map={"belief": "state"}, sync=Latest())

    # Monitor -> Planner (replan requests)
    pipe.connect(monitor, planner, map={"replan_config": "replan_config"}, sync=Latest())

    return pipe


def main():
    parser = argparse.ArgumentParser(description="Manipulation Pipeline Demo")
    parser.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    args = parser.parse_args()

    # Initialize retriever with global config
    retriever.init(
        backend="dora",
        backend_config={
            "dora_timeout": 10,
            "rerun_config": {"connect_addr": "127.0.0.1:9876"},
        }
    )

    # Build pipeline
    pipe = build_manipulation_pipeline()

    # Generate visualization before running
    from retriever.ir.viz import save_interactive_html
    save_interactive_html(pipe.build_ir(), "viz-manipulation-pipeline.html")

    # Run pipeline (backend_config from init() is used)
    pipe.run(duration=args.duration)


if __name__ == "__main__":
    main()
