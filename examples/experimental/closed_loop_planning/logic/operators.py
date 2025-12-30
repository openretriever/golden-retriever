from dataclasses import dataclass
from typing import List, Sequence, Set, Tuple

from retriever.types.options import ParameterizedOption
from retriever.types.symbolic import GroundAtom, LiftedAtom, Object, State, Variable

from ..types.domain import AtDoor, AtKey, Holding, IsOpen, door_var, key_var, robot_var
from ..types.options import Move, Pick, Unlock


@dataclass
class Operator:
    name: str
    parameters: Tuple[Variable, ...]
    preconditions: frozenset
    add_effects: frozenset
    delete_effects: frozenset
    option: ParameterizedOption

    def __init__(
        self,
        name: str,
        parameters: Sequence[Variable],
        preconditions: Set[LiftedAtom],
        add_effects: Set[LiftedAtom],
        delete_effects: Set[LiftedAtom],
        option: ParameterizedOption,
    ):
        self.name = name
        self.parameters = tuple(parameters)
        self.preconditions = frozenset(preconditions)
        self.add_effects = frozenset(add_effects)
        self.delete_effects = frozenset(delete_effects)
        self.option = option

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        params = ", ".join(v.name for v in self.parameters)
        return f"Operator<{self.name}({params})>"

@dataclass
class GroundOperator:
    operator: Operator
    grounding: Tuple[Object, ...]

    def get_option(self, state: State) -> ParameterizedOption:
        # Extract parameters from state based on operator type
        params: List[float] = []

        # Mapping for grounding lookup
        # mapping = dict(zip(self.operator.parameters, self.grounding))

        if self.operator.name == "MoveToKey":
            # grounding: (robot, key)
            key = self.grounding[1]
            if key in state:
                pos = state[key]
                params = [float(pos[0]), float(pos[1])]
            else:
                 raise KeyError(f"Key object {key} not found in Belief/State data during grounding.")

        elif self.operator.name == "MoveToDoor":
            # grounding: (robot, key, door) -> fixed to include key
            # or just (robot, door)??
            # Update: We changed definition below to (robot, key, door)
            # So door is at index 2
            door = self.grounding[2]
            if door in state:
                pos = state[door]
                # Door location in GridState is [x, y, open]
                params = [float(pos[0]), float(pos[1])]
            else:
                 raise KeyError(f"Door object {door} not found in Belief/State data during grounding.")

        # Pick and Unlock don't need scalar params in this domain (just objects)

        return self.operator.option.ground(self.grounding, params)

    def preconditions_grounded(self) -> Set[GroundAtom]:
        mapping = dict(zip(self.operator.parameters, self.grounding))
        return {a.ground(mapping) for a in self.operator.preconditions}

    def add_effects_grounded(self) -> Set[GroundAtom]:
        mapping = dict(zip(self.operator.parameters, self.grounding))
        return {a.ground(mapping) for a in self.operator.add_effects}

    def delete_effects_grounded(self) -> Set[GroundAtom]:
        mapping = dict(zip(self.operator.parameters, self.grounding))
        return {a.ground(mapping) for a in self.operator.delete_effects}

    def __repr__(self):
        args = ", ".join(o.name for o in self.grounding)
        return f"{self.operator.name}({args})"

    def __lt__(self, other):
        if not isinstance(other, GroundOperator):
            return NotImplemented
        return str(self) < str(other)


# --- Domain Operators ---

# MoveToKey: Robot moves from "Anywhere" to Key.
# Assumption: We don't track "At(Anywhere)" so we don't delete anything.
MoveToKeyOp = Operator(
    name="MoveToKey",
    parameters=(robot_var, key_var),
    preconditions=set(),
    add_effects={LiftedAtom(AtKey, (robot_var, key_var))},
    delete_effects=set(),
    option=Move,
)

# MoveToDoor (from Key): Robot moves from Key to Door.
# We explicitly parameterize 'key' so we can delete AtKey.
MoveToDoorOp = Operator(
    name="MoveToDoor",
    parameters=(robot_var, key_var, door_var),
    preconditions={LiftedAtom(AtKey, (robot_var, key_var))},
    add_effects={LiftedAtom(AtDoor, (robot_var, door_var))},
    delete_effects={LiftedAtom(AtKey, (robot_var, key_var))},
    option=Move,
)


# Pick should NOT delete AtKey (location matches). Only Move deletes location.
PickOp = Operator(
    name="Pick",
    parameters=(robot_var, key_var),
    preconditions={LiftedAtom(AtKey, (robot_var, key_var))},
    add_effects={LiftedAtom(Holding, (robot_var, key_var))},
    delete_effects=set(),
    option=Pick,
)

UnlockOp = Operator(
    name="Unlock",
    parameters=(robot_var, door_var, key_var), # Add key_var parameter
    # Require Holding key and AtDoor
    preconditions={
        LiftedAtom(AtDoor, (robot_var, door_var)),
        LiftedAtom(Holding, (robot_var, key_var))
    },
    add_effects={LiftedAtom(IsOpen, (door_var,))},
    delete_effects=set(),
    option=Unlock,
)

GRID_OPERATORS = {MoveToKeyOp, MoveToDoorOp, PickOp, UnlockOp}
