from typing import Optional, Type

import numpy as np

from retriever.flow import Flow
from retriever.types.options import Action, Option

# Import types from correct location
from ..types.flow_types import ExecutorInput, ExecutorOutput


class SkillExecutorFlow(Flow[ExecutorInput, ExecutorOutput]):
    """Skill Execution layer implementing MPC-style option execution."""

    name: str = "skill_executor"
    input_type: Type = ExecutorInput
    output_type: Type = ExecutorOutput

    def __init__(self, name: str = "skill_executor"):
        super().__init__()
        self.name = name
        self.current_option: Optional[Option] = None
        # Rerun initialized globally

    def step(self, inp: ExecutorInput) -> ExecutorOutput:
        import rerun as rr
        state = inp.state
        if state is None:
            return ExecutorOutput(action=None, status="waiting_for_state")

        noop_action = Action(arr=np.array([0.0, 0.0, 0.0]))
        plan = inp.plan

        if not plan:
            rr.log("planning/current_option", rr.TextDocument("None"))
            return ExecutorOutput(action=noop_action, status="success")

        target_option = plan[0]

        # Check if we need to switch option
        should_switch = True
        if self.current_option:
            if (
                self.current_option.name == target_option.name
                and self.current_option.objects == target_option.objects
                and np.allclose(
                    np.array(self.current_option.params),
                    np.array(target_option.params),
                )
            ):
                should_switch = False

        if should_switch:
            print(f"EXECUTOR: Switching to option '{target_option.name}'")
            rr.log("planning/current_option", rr.TextDocument(f"Switching: {target_option.name}"))
            if not target_option.initiable(state):
                print(f"EXECUTOR: Precondition violated for '{target_option.name}'!")
                rr.log("planning/error", rr.TextDocument(f"Precondition failed: {target_option.name}"))
                return ExecutorOutput(action=noop_action, status="failure")
            self.current_option = target_option

        # Check termination
        if self.current_option.terminal(state):
            print(f"[{self.name}] Option {self.current_option.name} TERMINATED successfully.")
            rr.log("planning/current_option", rr.TextDocument(f"Terminated: {self.current_option.name}"))
            self.current_option = None
            return ExecutorOutput(action=noop_action, status="success")

        # Execute policy
        if self.current_option:
            try:
                # Debugging RISE pipeline
                robot_obj = self.current_option.objects[0]
                robot_pos = state.get(robot_obj)
                print(f"[{self.name}] Option: {self.current_option.name}, Params: {self.current_option.params}, RobotPos: {robot_pos}")
            except Exception as e:
                print(f"[{self.name}] Debug Error: {e}")

        action = self.current_option.policy(state)
        action.set_option(self.current_option)
        return ExecutorOutput(action=action, status="in_progress")
