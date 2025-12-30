from retriever.flow import Latest, Pipeline, Rate, Trigger

from ..flows.belief_updater import BeliefUpdaterFlow

# Flows
from ..flows.env_rise import RiseEnvironmentFlow
from ..flows.monitor_execution import ExecutionMonitorFlow
from ..flows.perception import PerceptionFlow
from ..flows.planner_astar import TaskPlannerFlow
from ..flows.skill_executor import SkillExecutorFlow

# Data Types


def build_rise_pipeline() -> Pipeline:
    # 2. Define Pipeline
    p = Pipeline("rise_complete")

    # 1. Initialize Flows with Triggers
    rise_env = RiseEnvironmentFlow(name="RiseEnvironmentFlow") @ Rate(10.0)

    # Perception triggers on sensor data arrival
    perception = PerceptionFlow(name="PerceptionFlow") @ Trigger("data")

    # Belief updates on new observation state
    belief_updater = BeliefUpdaterFlow(name="BeliefUpdaterFlow") @ Trigger("observation")

    # Planner runs when belief changes
    planner = TaskPlannerFlow(name="TaskPlannerFlow") @ Trigger("state")

    # Executor runs when belief (state) or plan changes
    # Triggers on 'state' (belief_for_execution) input
    executor = SkillExecutorFlow(name="SkillExecutorFlow") @ Trigger("state")

    # Monitor runs when executor status updates
    monitor = ExecutionMonitorFlow(name="ExecutionMonitorFlow") @ Trigger("executor_status")

    # 3. Add Nodes - Implicitly handled by connect() in recent versions, or use p.add(flow)?
    # complete.py DOES NOT call add_node. It just defines flows and connects.
    # We will trust connect() or variable scope capture is enough (or rather, connect registers them).

    # 4. Connect Flows
    # Note: Using Latest() for synchronization where appropriate, mimicking complete.py

    # Environment -> Perception
    # EnvOutput.data -> PerceptionInput.data
    p.connect(rise_env, perception, map={"data": "data"}, sync=Latest())

    # Perception -> Belief
    # PerceptionOutput.state -> BeliefUpdateInput.observation
    # PerceptionOutput.atoms -> BeliefUpdateInput.visible_atoms
    p.connect(perception, belief_updater, map={"state": "observation", "atoms": "visible_atoms"}, sync=Latest())

    # Executor -> Belief (Action History)
    p.connect(executor, belief_updater, map={"action": "action"}, sync=Latest())

    from ..flows.debug_flow import DebugFlow
    debug_node = DebugFlow() @ Trigger("data")
    # p.add_node(debug_node)  <-- Removed
    p.connect(belief_updater, debug_node, map={"belief": "data"}, sync=Latest())

    # Task Planner
    # planner = TaskPlannerFlow(name="TaskPlannerFlow") # Already defined above
    # p.add_node(planner) <--- Removed

    # Task Planner
    # planner = TaskPlannerFlow(name="TaskPlannerFlow") @ Trigger("state") # Defined above

    # Belief -> Planner
    # BeliefUpdateOutput.belief -> PlannerInput.state
    p.connect(belief_updater, planner, map={"belief": "state"}, sync=Latest())

    # Planner -> Executor
    # PlannerOutput.plan -> ExecutorInput.plan
    p.connect(planner, executor, map={"plan": "plan"}, sync=Latest())

    # Belief -> Executor (Grounding context)
    p.connect(belief_updater, executor, map={"belief": "state"}, sync=Latest())

    # Executor -> Env
    # ExecutorOutput.action -> EnvInput.action
    p.connect(executor, rise_env, map={"action": "action"}, sync=Latest())

    # Monitor Loop
    p.connect(executor, monitor, map={"status": "executor_status"}, sync=Latest())
    p.connect(belief_updater, monitor, map={"belief": "state"}, sync=Latest())

    # Monitor -> Planner (Replan request)
    p.connect(monitor, planner, map={"replan_config": "replan_config"}, sync=Latest())

    return p

def main():
    import argparse

    from retriever.ir.viz import save_interactive_html

    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    pipeline = build_rise_pipeline()

    # Save visualization
    save_interactive_html(pipeline.build_ir(), "rise_pipeline_viz.html")

    import os
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    
    print(f"Starting RISE Pipeline... (Key present: {bool(gemini_key)})")
    pipeline.run(
        backend="dora",
        duration=args.duration,
        backend_config={
            "dora_timeout": 10,
            "rerun_config": {"spawn": True, "connect_addr": "127.0.0.1:9876"},
            "env_overrides": {
                "GEMINI_API_KEY": gemini_key
            }
        }
    )
    print("RISE Pipeline Finished.")

if __name__ == "__main__":
    main()
