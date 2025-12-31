# Closed Loop Planning Pipeline (Simple/Heuristic)
#
# Environment → Perception → BeliefUpdater → Planner → Executor → Environment

import argparse

from retriever.flow import Latest, Pipeline, Rate, Trigger
from retriever.types.options import Task
from retriever.types.symbolic import GroundAtom, Object, State

from ..flows.belief_updater import BeliefUpdaterFlow

# Import Flows
from ..flows.env_simple_grid import GridEnvironmentFlow
from ..flows.planner_heuristic import PlannerFlow
from ..flows.monitor_execution import ExecutionMonitorFlow
from ..flows.perception import PerceptionFlow
from ..flows.skill_executor import SkillExecutorFlow
from ..types.domain import IsOpen, door_type
from retriever.ir.viz import save_interactive_html


def build_simple_pipeline() -> Pipeline:
    """Build the closed-loop planning pipeline."""
    door_obj = Object("door", door_type)
    goal_atom = GroundAtom(IsOpen, [door_obj])
    # task = Task(init=State({}), goal={goal_atom}) # Task is initialized inside PlannerFlow if None

    pipe = Pipeline("closed_loop_simple")

    save_interactive_html(pipe.build_ir(), "viz-simple-pipeline.html")

    # Instantiate Flows with Decorators
    env = GridEnvironmentFlow() @ Rate(10.0)

    perception = PerceptionFlow() @ Trigger("data")

    belief = BeliefUpdaterFlow() @ Trigger("observation")

    planner = PlannerFlow(name="PlannerFlow") @ Trigger("replan_config")

    executor = SkillExecutorFlow("skill_executor") @ Trigger("state")

    monitor = ExecutionMonitorFlow() @ Trigger("executor_status")

    # Wiring

    # Env -> Perception
    pipe.connect(env, perception, map={"data": "data"}, sync=Latest())

    # Perception -> BeliefUpdater
    pipe.connect(perception, belief, map={"state": "observation", "atoms": "visible_atoms"}, sync=Latest())

    # Executor -> Belief
    pipe.connect(executor, belief, map={"action": "action"}, sync=Latest())

    # BeliefUpdater -> Planner, Executor, Monitor
    pipe.connect(belief, planner, map={"belief": "state"}, sync=Latest())
    pipe.connect(belief, executor, map={"belief": "state"}, sync=Latest())
    pipe.connect(belief, monitor, map={"belief": "state"}, sync=Latest())

    # Planner -> Executor
    pipe.connect(planner, executor, map={"plan": "plan"}, sync=Latest())

    # Executor -> Monitor
    pipe.connect(executor, monitor, map={"status": "executor_status"}, sync=Latest())

    # Monitor -> Planner
    pipe.connect(monitor, planner, map={"replan_config": "replan_config"}, sync=Latest())

    # Executor -> Env
    pipe.connect(executor, env, map={"action": "action"}, sync=Latest())

    return pipe

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=float, default=5.0)
    args = parser.parse_args()

    pipe = build_simple_pipeline()
    pipe.run(
        duration=args.duration,
        backend="dora",
        backend_config={
            "dora_timeout": 10,
            "rerun_config": {"spawn": True, "connect_addr": "127.0.0.1:9876"}
        }
    )
