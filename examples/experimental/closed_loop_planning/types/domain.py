from typing import Sequence

import numpy as np

from retriever.types.symbolic import Object, ObjectType, Predicate, State, Variable

# --- Types ---
robot_type = ObjectType("robot", ["x", "y", "grip"])
key_type = ObjectType("key", ["x", "y", "held"])
door_type = ObjectType("door", ["x", "y", "is_open"])

# --- Variables ---
robot_var = Variable("?robot", robot_type)
key_var = Variable("?key", key_type)
door_var = Variable("?door", door_type)

# --- Objects (Singletons for simple demo) ---
robot_obj = Object("robot", robot_type)
key_obj = Object("key", key_type)
door_obj = Object("door", door_type)

# --- Predicates ---


def _at_classifier(state: State, objects: Sequence[Object]) -> bool:
    # Generic proximity check
    # Assumes objects have [x, y, ...] as first 2 features
    a, b = objects
    pos_a = state[a][:2]
    pos_b = state[b][:2]
    return np.linalg.norm(pos_a - pos_b) < 0.1


AtKey = Predicate("AtKey", (robot_type, key_type), _at_classifier)
AtDoor = Predicate("AtDoor", (robot_type, door_type), _at_classifier)


def _holding_classifier(state: State, objects: Sequence[Object]) -> bool:
    robot, key = objects
    # Check if key is held (feature 2 > 0.5)
    return state[key][2] > 0.5


Holding = Predicate("Holding", (robot_type, key_type), _holding_classifier)



# VLM Predicate
from ..vlm import VisualPredicate


# Geometric Fallback (used internally by VisualPredicate if image missing)
def _open_classifier(state: State, objects: Sequence[Object]) -> bool:
    door = objects[0]
    return state[door][2] > 0.5

IsOpen = VisualPredicate(
    name="IsOpen",
    types=(door_type,),
    prompt="Is the door open?"
)
# Monkey-patch fallback for when VLM is disabled/fails
IsOpen._geometric_fallback = _open_classifier
