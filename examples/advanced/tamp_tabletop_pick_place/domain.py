from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

Atom = tuple[str, tuple[str, ...]]
State = frozenset[Atom]
Binding = dict[str, str]


def atom(predicate: str, *args: str) -> Atom:
    return (predicate, tuple(args))


def _is_variable(token: str) -> bool:
    return token.startswith("?")


@dataclass(frozen=True, order=True)
class GroundAction:
    name: str
    args: tuple[str, ...]

    def __str__(self) -> str:
        return f"{self.name}({', '.join(self.args)})"


@dataclass(frozen=True)
class Operator:
    name: str
    parameters: tuple[tuple[str, str], ...]
    preconditions: tuple[Atom, ...]
    add_effects: tuple[Atom, ...]
    delete_effects: tuple[Atom, ...]

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

    def grounded_preconditions(self, binding: Mapping[str, str]) -> tuple[Atom, ...]:
        return tuple(_ground_atom(pattern, binding) for pattern in self.preconditions)

    def grounded_add_effects(self, binding: Mapping[str, str]) -> tuple[Atom, ...]:
        return tuple(_ground_atom(pattern, binding) for pattern in self.add_effects)

    def grounded_delete_effects(self, binding: Mapping[str, str]) -> tuple[Atom, ...]:
        return tuple(_ground_atom(pattern, binding) for pattern in self.delete_effects)

    def is_applicable(self, state: State, binding: Mapping[str, str]) -> bool:
        return set(self.grounded_preconditions(binding)).issubset(state)

    def apply(self, state: State, binding: Mapping[str, str]) -> State:
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

DEFAULT_INITIAL_STATE: State = frozenset(
    {
        atom("HandEmpty"),
        atom("InRegion", "red_block", "start_region"),
    }
)

DEFAULT_GOAL_ATOMS: State = frozenset(
    {
        atom("InRegion", "red_block", "goal_region"),
    }
)


def _ground_atom(pattern: Atom, binding: Mapping[str, str]) -> Atom:
    predicate, args = pattern
    grounded_args = tuple(binding.get(arg, arg) if _is_variable(arg) else arg for arg in args)
    return (predicate, grounded_args)


def bind_action(action: GroundAction) -> Binding:
    operator = OPERATORS_BY_NAME[action.name]
    return {
        variable_name: object_name
        for (variable_name, _), object_name in zip(operator.parameters, action.args)
    }


def apply_ground_action(state: State, action: GroundAction) -> State:
    operator = OPERATORS_BY_NAME[action.name]
    binding = bind_action(action)
    return operator.apply(state, binding)


def goals_satisfied(state: State, goal_atoms: State = DEFAULT_GOAL_ATOMS) -> bool:
    return set(goal_atoms).issubset(state)


def action_signature(action: GroundAction) -> str:
    return str(action)


def format_atoms(atoms: Iterable[Atom]) -> str:
    rendered = []
    for predicate, args in sorted(atoms):
        rendered.append(f"{predicate}({', '.join(args)})")
    return "{" + ", ".join(rendered) + "}"


def pretty_state(state: State) -> str:
    return format_atoms(state)
