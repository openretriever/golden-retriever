from typing import Optional

from retriever.flow import Flow
from retriever.types.options import Task

from ..types.domain import (
    AtDoor,
    AtKey,
    Holding,
    IsOpen,
    door_type,
    key_type,
    robot_type,
)
from ..types.flow_types import PlannerInput, PlannerOutput
from ..types.options import Move, Pick, Unlock


class PlannerFlow(Flow[PlannerInput, PlannerOutput]):
    """Heuristic planner for the closed-loop POC."""

    def __init__(self, task: Optional[Task] = None, name: str = "PlannerFlow"):
        self.name = name
        if task is None:
            from retriever.types.symbolic import GroundAtom, Object, State

            from ..types.domain import IsOpen, door_type

            door_obj = Object("door", door_type)
            goal_atom = GroundAtom(IsOpen, [door_obj])
            task = Task(init=State({}), goal={goal_atom})
        self.task = task

    def step(self, inp: PlannerInput) -> PlannerOutput:
        if inp.replan_config and not inp.replan_config.should_replan:
            return PlannerOutput(plan=[])

        state = inp.state
        if state is None:
            return PlannerOutput(plan=[])

        self.task.init = state

        # Extract objects from state
        robot = next(o for o in state.data if o.type == robot_type)
        key = next(o for o in state.data if o.type == key_type)
        door = next(o for o in state.data if o.type == door_type)

        plan = []
        has_key = Holding.holds(state, [robot, key])

        if not has_key:
            # Phase 1: Get Key
            if not AtKey.holds(state, [robot, key]):
                target_pos = state[key][:2]
                plan.append(Move.ground([robot], list(target_pos)))
            plan.append(Pick.ground([robot, key], []))

            # Lookahead: After Pick, go to Door
            door_pos = state[door][:2]
            plan.append(Move.ground([robot], list(door_pos)))
            plan.append(Unlock.ground([robot, door, key], []))
        else:
            # Phase 2: Open Door
            door_open = IsOpen.holds(state, [door])
            if not door_open:
                if not AtDoor.holds(state, [robot, door]):
                    target_pos = state[door][:2]
                    plan.append(Move.ground([robot], list(target_pos)))
                plan.append(Unlock.ground([robot, door, key], []))

        if plan:
            print(f"[PlannerFlow] New plan with {len(plan)} options.")

        return PlannerOutput(plan=plan)
