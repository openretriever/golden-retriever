# Closed Loop Planning Pipeline (Complete)
#
# Integration of:
# 1. GridEnvironment: Simulates robot, key, door, and discrete actions.
# 2. Perception: Simulates visual processing to produce symbolic atoms.
# 3. BeliefUpdater: Maintains epistemic state (Known/Unknown) and belief history.
# 4. TaskPlanner (A*): Generates high-level plans using domain operators and belief state.
# 5. SkillExecutor: Executes plan options using MPC-style geometric policies.
# 6. Monitor: Tracks execution status and triggers replanning on failure.

from retriever.flow import Latest, Pipeline, Rate, Trigger
from retriever.ir.viz import save_interactive_html
from retriever.types.symbolic import GroundAtom, Object

from ..flows.belief_updater import BeliefUpdaterFlow

# --- Flows ---
from ..flows.monitor_execution import ExecutionMonitorFlow
from ..flows.perception import PerceptionFlow
from ..flows.planner_astar import TaskPlannerFlow
from ..flows.skill_executor import SkillExecutorFlow

# --- Types & Logic ---
from ..types.domain import IsOpen, door_type


def build_complete_pipeline():
    """Builds and configures the complete closed-loop planning pipeline."""

    # 1. Pipeline Configuration
    pipeline_name = "closed_loop_complete"
    door = Object("door", door_type)
    goal_atoms = {GroundAtom(IsOpen, (door,))}

    pipe = Pipeline(pipeline_name)

    # 2. Flow Instantiation & Rate Configuration
    # - Environment runs at 10Hz to simulate physics/world state.
    env = RiseEnvironmentFlow("grid_env") @ Rate(10.0)

    # - Perception runs when Environment emits data.
    perception = PerceptionFlow("perception") @ Trigger("data")

    # - BeliefUpdater runs on new observations to maintain state estimate.
    belief = BeliefUpdaterFlow() @ Trigger("observation")

    # - Planner runs when triggered by Belief updates (reactive) or Monitor (replan).
    #   Initialized with domain operators and static objects.
    planner_inst = TaskPlannerFlow(name="task_planner")
    planner = planner_inst @ Trigger("state")

    # - Executor runs on State updates to compute control actions (MPC loop).
    executor = SkillExecutorFlow("skill_executor") @ Trigger("state")

    # - Monitor checks status updates from Executor.
    monitor = ExecutionMonitorFlow("monitor") @ Trigger("executor_status")

    # 3. Wiring (Data Flow)

    # Perception Loop
    pipe.connect(env, perception, map={"data": "data"}, sync=Latest())
    pipe.connect(perception, belief, map={"state": "observation", "atoms": "visible_atoms"}, sync=Latest())

    # Action Feedback (Loop Closure)
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

    # Actuation
    pipe.connect(executor, env, map={"action": "action"}, sync=Latest())

    return pipe

if __name__ == "__main__":
    # Run for a fixed duration to demonstrate the full task sequence:
    # Move -> Pick Key -> Move -> Unlock Door
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=10.0)
    args = parser.parse_args()

    pipe = build_complete_pipeline()
    save_interactive_html(pipe.build_ir(), "closed_loop_pipeline_complete_demo_viz.html")

    pipe.run(
        duration=args.duration,
        backend="dora",
        backend_config={
            "dora_timeout": 10,
            "rerun_config": {"spawn": True, "connect_addr": "127.0.0.1:9876"}
        }
    )
