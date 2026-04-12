from __future__ import annotations

import heapq
from itertools import count
from typing import Sequence

from retriever_tamp.core.types import GroundAction, SymbolicState

from domain import (
    DEFAULT_GOAL_ATOMS,
    DEFAULT_INITIAL_STATE,
    OBJECT_SETS,
    OPERATORS,
    action_signature,
    apply_ground_action,
)


def get_applicable_actions(
    state: SymbolicState,
    *,
    banned_action_signatures: frozenset[str] = frozenset(),
) -> list[GroundAction]:
    actions: list[GroundAction] = []
    for operator in OPERATORS:
        for binding in operator.iter_bindings(OBJECT_SETS):
            action = operator.ground_action(binding)
            if action_signature(action) in banned_action_signatures:
                continue
            if operator.is_applicable(state, binding):
                actions.append(action)
    return sorted(actions)


def _heuristic(state: SymbolicState, goal_atoms: SymbolicState) -> int:
    return len(set(goal_atoms) - set(state))


def task_plan(
    initial_state: SymbolicState = DEFAULT_INITIAL_STATE,
    goal_atoms: SymbolicState = DEFAULT_GOAL_ATOMS,
    *,
    banned_action_signatures: frozenset[str] = frozenset(),
    max_expansions: int = 64,
) -> list[GroundAction]:
    frontier: list[tuple[int, int, int, SymbolicState, list[GroundAction]]] = []
    ticket = count()
    heapq.heappush(
        frontier,
        (_heuristic(initial_state, goal_atoms), 0, next(ticket), initial_state, []),
    )

    best_cost_by_state: dict[SymbolicState, int] = {initial_state: 0}
    expansions = 0

    while frontier and expansions < max_expansions:
        _, g_cost, _, state, plan = heapq.heappop(frontier)
        expansions += 1

        if set(goal_atoms).issubset(state):
            return plan

        for action in get_applicable_actions(
            state,
            banned_action_signatures=banned_action_signatures,
        ):
            next_state = apply_ground_action(state, action)
            next_cost = g_cost + 1
            if best_cost_by_state.get(next_state, 10**9) <= next_cost:
                continue
            best_cost_by_state[next_state] = next_cost
            priority = next_cost + _heuristic(next_state, goal_atoms)
            heapq.heappush(
                frontier,
                (priority, next_cost, next(ticket), next_state, plan + [action]),
            )

    return []


def format_plan(plan: Sequence[GroundAction]) -> str:
    if not plan:
        return "<no plan>"
    return " -> ".join(str(action) for action in plan)
