# Task Planning Infrastructure
#
# Minimal STRIPS-style task planning for closed-loop robotics.

import heapq
from typing import List, Optional, Sequence, Set, Tuple

from rich import print as rprint

from retriever.flow import Flow
from retriever.types.options import Option
from retriever.types.symbolic import GroundAtom, Object, Variable

from ..logic.operators import GRID_OPERATORS, GroundOperator, Operator
from ..types.belief import BeliefState
from ..types.domain import IsOpen, door_obj, key_obj, robot_obj
from ..types.flow_types import PlannerInput, PlannerOutput


def get_applicable_operators(
    state_atoms: Set[GroundAtom],
    operators: Sequence[Operator],
    objects: Set[Object]
) -> List[GroundOperator]:
    """Find all operators applicable in current state."""
    applicable = []
    for op in operators:
        for grounding in _get_all_groundings(op.parameters, objects):
            ground_op = GroundOperator(op, grounding)
            preconds = ground_op.preconditions_grounded()
            if preconds.issubset(state_atoms):
                applicable.append(ground_op)
    return applicable

def apply_operator(
    state_atoms: Set[GroundAtom],
    ground_op: GroundOperator
) -> Set[GroundAtom]:
    """Apply operator to get successor state atoms."""
    new_atoms = state_atoms.copy()
    new_atoms -= ground_op.delete_effects_grounded()
    new_atoms |= ground_op.add_effects_grounded()
    return new_atoms

def task_plan_astar(
    init_atoms: Set[GroundAtom],
    goal_atoms: Set[GroundAtom],
    operators: Sequence[Operator],
    objects: Set[Object],
    max_steps: int = 100,
) -> Optional[List[GroundOperator]]:
    """A* task planner."""
    h0 = len(goal_atoms - init_atoms)
    start = (h0, 0, frozenset(init_atoms), [])
    frontier = [start]

    # Track visited states to avoid cycles
    visited: Set[frozenset] = set()

    steps = 0
    while frontier and steps < max_steps:
        steps += 1
        _, g_cost, state_fs, plan = heapq.heappop(frontier)
        if state_fs in visited:
            continue
        visited.add(state_fs)

        state_atoms = set(state_fs)
        if goal_atoms.issubset(state_atoms):
            print(f"DEBUG: A* found goal at step {g_cost}")
            return plan

        applicable = get_applicable_operators(state_atoms, operators, objects)
        print(f"DEBUG: Step {steps}. State={state_atoms}. Applicable Ops={len(applicable)}")
        for op in applicable:
            print(f"  - {op}")

        for ground_op in applicable:
            new_atoms = apply_operator(state_atoms, ground_op)
            new_fs = frozenset(new_atoms)
            if new_fs not in visited:
                h = len(goal_atoms - new_atoms)
                new_g = g_cost + 1
                heapq.heappush(frontier, (new_g + h, new_g, new_fs, plan + [ground_op]))

    return None

def _get_all_groundings(
    params: Sequence[Variable],
    objects: Set[Object]
) -> List[Tuple[Object, ...]]:
    if not params:
        return [()]
    first, rest = params[0], params[1:]
    result = []
    for obj in objects:
        if obj.is_instance(first.type):
            for rest_grounding in _get_all_groundings(rest, objects):
                result.append((obj,) + rest_grounding)
    return result

# --- Flow Interface ---

class TaskPlannerFlow(Flow[PlannerInput, PlannerOutput]):
    """Task planner as a Retriever Flow."""

    def __init__(self, name: str = "TaskPlanner"):
        self.name = name
        self.current_plan: List[Option] = []

        # Rerun initialized globally by DoraExecutor

        # Load Domain Configuration directly to avoid serialization issues
        # in Dora/multiprocessing contexts with complex objects.
        self.operators = list(GRID_OPERATORS)
        self.objects = {robot_obj, key_obj, door_obj}

        # Define Goal: Open the door
        # In a real system, this might be passed via context or task input.
        # Here we hardcode for the demo.
        door = [o for o in self.objects if o.type.name == "door"][0]
        self.goal_atoms = {GroundAtom(IsOpen, [door])}

    def step(self, inp: PlannerInput) -> PlannerOutput:
        rprint(f"[{self.name}] Step called. Belief={'[green]Present[/]' if inp.state else '[red]None[/]'} (from .state)")

        belief = inp.state
        # If no belief, we can't plan (unless start state empty, but we assume belief is required)
        if belief is None:
            return PlannerOutput(plan=[], success=False)

        print(f"[{self.name}] Planning for belief: {belief}")

        # Check if we need to replan
        should_replan = False
        if not self.current_plan:
            should_replan = True
        elif inp.replan_config and inp.replan_config.should_replan:
             should_replan = True

        if not should_replan:
            return PlannerOutput(plan=self.current_plan)

        # Extract atoms from input or belief
        current_atoms: Set[GroundAtom] = set()

        # In current Pipeline, we map 'belief' -> 'state', so likely `inp.state` has what we need.
        # PlannerInput has 'state': BeliefState.
        # BeliefState has 'epistemic.known_true' which are the atoms.
        if inp.state and inp.state.epistemic:
             current_atoms.update(inp.state.epistemic.known_true)

        print(f"[{self.name}] Planning goal={self.goal_atoms}")
        print(f"[{self.name}] Current Atoms: {current_atoms}")

        ground_plan = task_plan_astar(
            init_atoms=current_atoms,
            goal_atoms=self.goal_atoms,
            operators=self.operators,
            objects=self.objects,
        )

        if ground_plan is None:
            print(f"[{self.name}] A* failed to find plan.")
            return PlannerOutput(plan=[], success=False)

        # Convert Ground Operators to Options
        try:
            # We must pass the BeliefState to get_option for parameter grounding
            current_state = inp.state if inp.state else BeliefState(data={})

            self.current_plan = [g_op.get_option(current_state) for g_op in ground_plan]

            plan_str = " -> ".join([op.operator.name for op in ground_plan])
            print(f"[{self.name}] PLAN FOUND: {plan_str}")

        except Exception as e:
             import traceback
             print(f"[{self.name}] Failed to ground options: {e}")
             traceback.print_exc()
             return PlannerOutput(plan=[], success=False)

        return PlannerOutput(plan=self.current_plan, success=True)
