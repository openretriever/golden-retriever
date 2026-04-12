from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

from retriever_tamp.core.types import GroundAction, GroundAtom, SymbolicState

Binding = dict[str, str]


def atom(predicate: str, *args: str) -> GroundAtom:
    return GroundAtom(predicate=predicate, args=tuple(args))


def _is_variable(token: str) -> bool:
    return token.startswith("?")

@dataclass(frozen=True)
class Operator:
    name: str
    parameters: tuple[tuple[str, str], ...]
    preconditions: tuple[GroundAtom, ...]
    add_effects: tuple[GroundAtom, ...]
    delete_effects: tuple[GroundAtom, ...]

    def iter_bindings(self, object_sets: Mapping[str, Sequence[str]]) -> Iterable[Binding]:
        if not self.parameters:
            yield {}
            return

        choices = [object_sets[type_name] for _, type_name in self.parameters]
        for grounded_objects in product(*choices):
            yield {
                variable_name: object_name
                for (variable_name, _), object_name in zip(self.parameters, grounded_objects)
            }

    def ground_action(self, binding: Mapping[str, str]) -> GroundAction:
        args = tuple(binding[variable_name] for variable_name, _ in self.parameters)
        return GroundAction(self.name, args)

    def grounded_preconditions(self, binding: Mapping[str, str]) -> tuple[GroundAtom, ...]:
        return tuple(_ground_atom(pattern, binding) for pattern in self.preconditions)

    def grounded_add_effects(self, binding: Mapping[str, str]) -> tuple[GroundAtom, ...]:
        return tuple(_ground_atom(pattern, binding) for pattern in self.add_effects)

    def grounded_delete_effects(self, binding: Mapping[str, str]) -> tuple[GroundAtom, ...]:
        return tuple(_ground_atom(pattern, binding) for pattern in self.delete_effects)

    def is_applicable(self, state: SymbolicState, binding: Mapping[str, str]) -> bool:
        return set(self.grounded_preconditions(binding)).issubset(state)

    def apply(self, state: SymbolicState, binding: Mapping[str, str]) -> SymbolicState:
        next_atoms = set(state)
        next_atoms -= set(self.grounded_delete_effects(binding))
        next_atoms |= set(self.grounded_add_effects(binding))
        return frozenset(next_atoms)


OBJECT_SETS: dict[str, tuple[str, ...]] = {
    "object": ("red_block",),
    "region": ("start_region", "goal_region"),
}

PICK = Operator(
    name="Pick",
    parameters=(("?obj", "object"), ("?region", "region")),
    preconditions=(
        atom("HandEmpty"),
        atom("InRegion", "?obj", "?region"),
    ),
    add_effects=(atom("Holding", "?obj"),),
    delete_effects=(
        atom("HandEmpty"),
        atom("InRegion", "?obj", "?region"),
    ),
)

PLACE = Operator(
    name="Place",
    parameters=(("?obj", "object"), ("?region", "region")),
    preconditions=(atom("Holding", "?obj"),),
    add_effects=(
        atom("HandEmpty"),
        atom("InRegion", "?obj", "?region"),
    ),
    delete_effects=(atom("Holding", "?obj"),),
)

OPERATORS: tuple[Operator, ...] = (PICK, PLACE)
OPERATORS_BY_NAME = {operator.name: operator for operator in OPERATORS}

DEFAULT_INITIAL_STATE: SymbolicState = frozenset(
    {
        atom("HandEmpty"),
        atom("InRegion", "red_block", "start_region"),
    }
)

DEFAULT_GOAL_ATOMS: SymbolicState = frozenset(
    {
        atom("InRegion", "red_block", "goal_region"),
    }
)


def _ground_atom(pattern: GroundAtom, binding: Mapping[str, str]) -> GroundAtom:
    grounded_args = tuple(
        binding.get(arg, arg) if _is_variable(arg) else arg for arg in pattern.args
    )
    return GroundAtom(predicate=pattern.predicate, args=grounded_args)


def bind_action(action: GroundAction) -> Binding:
    operator = OPERATORS_BY_NAME[action.name]
    return {
        variable_name: object_name
        for (variable_name, _), object_name in zip(operator.parameters, action.args)
    }


def apply_ground_action(state: SymbolicState, action: GroundAction) -> SymbolicState:
    operator = OPERATORS_BY_NAME[action.name]
    binding = bind_action(action)
    return operator.apply(state, binding)


def goals_satisfied(
    state: SymbolicState, goal_atoms: SymbolicState = DEFAULT_GOAL_ATOMS
) -> bool:
    return set(goal_atoms).issubset(state)


def action_signature(action: GroundAction) -> str:
    return str(action)


def format_atoms(atoms: Iterable[GroundAtom]) -> str:
    return "{" + ", ".join(str(atom) for atom in sorted(atoms)) + "}"


def pretty_state(state: SymbolicState) -> str:
    return format_atoms(state)
