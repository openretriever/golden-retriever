# Closed Loop Planning Pipeline (Simulated / Real-Image Synthetic Env)
#
# Same structure as complete.py/simple.py but uses the RiseEnvironmentFlow
# which mocks real robot interfaces more closely than GridEnvironment.

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger

from ..flows.belief_updater import BeliefUpdaterFlow

# --- Flows ---
# Using SpotEnvironmentFlow for Real Robot
from ..flows.env_spot import SpotEnvironmentFlow
from ..flows.monitor_execution import ExecutionMonitorFlow
from ..flows.perception import PerceptionFlow
from ..flows.planner_astar import TaskPlannerFlow
from ..flows.skill_executor import SkillExecutorFlow
from retriever.ir.viz import save_interactive_html


def build_spot_pipeline() -> Pipeline:
    """Builds the closed-loop planning pipeline for REAL SPOT robot."""

    # 1. Pipeline Configuration
    pipeline_name = "spot_real_pipeline"

    pipe = Pipeline(pipeline_name)

    # 2. Flow Instantiation
    # - Environment (Spot) runs at 10Hz
    env = SpotEnvironmentFlow("spot_driver") @ Rate(10.0)

    # - Perception runs when Environment emits data (images/state)
    perception = PerceptionFlow("perception") @ Trigger("data")

    # - BeliefUpdater maintains state
    belief = BeliefUpdaterFlow() @ Trigger("observation")

    # - Planner (A*)
    planner = TaskPlannerFlow("task_planner") @ Trigger("state")

    # - Executor (MPC / Skills)
    executor = SkillExecutorFlow("skill_executor") @ Trigger("state")

    # - Monitor
    monitor = ExecutionMonitorFlow("monitor") @ Trigger("executor_status")

    # 3. Wiring

    # Perception Loop
    pipe.connect(env, perception, map={"data": "data"}, sync=Latest())
    pipe.connect(perception, belief, map={"state": "observation", "atoms": "visible_atoms"}, sync=Latest())

    # Action Feedback
    pipe.connect(executor, belief, map={"action": "action"}, sync=Latest())

    # Planning & Execution
    pipe.connect(belief, planner, map={"belief": "state"}, sync=Latest())
    pipe.connect(monitor, planner, map={"replan_config": "replan_config"}, sync=Latest())
    pipe.connect(planner, executor, map={"plan": "plan"}, sync=Latest())

    # State Distribution
    pipe.connect(belief, executor, map={"belief": "state"}, sync=Latest())
    pipe.connect(belief, monitor, map={"belief": "state"}, sync=Latest())

    # Status Monitoring
    pipe.connect(executor, monitor, map={"status": "executor_status"}, sync=Latest())

    # Actuation (Command to Spot)
    pipe.connect(executor, env, map={"action": "action"}, sync=Latest())

    return pipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    print("Starting Spot Pipeline (Real Robot)...")
    pipe = build_spot_pipeline()
    save_interactive_html(pipe.build_ir(), "viz-spot-pipeline.html")

    # Run with Dora backend
    # Note: Requires BOSDYN env vars to be set
    pipe.run(
        backend="dora",
        duration=args.duration,
        backend_config={
            "dora_timeout": 10,
            "rerun_config": {"spawn": True, "connect_addr": "127.0.0.1:9876"}
        }
    )
