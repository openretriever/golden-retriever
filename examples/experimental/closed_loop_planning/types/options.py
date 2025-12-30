from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np

from retriever.types.options import Action, ParameterizedOption
from retriever.types.symbolic import Object, State

from .domain import AtDoor, AtKey, Holding, IsOpen, door_type, key_type, robot_type


@dataclass
class ReplanConfig:
    should_replan: bool = False
    reason: str = ""


# --- Policies & Conditions ---


def move_policy(
    state: State, memory: Dict, objects: Sequence[Object], params: Sequence[float]
) -> Action:
    # params: [target_x, target_y]
    # For Move(robot), params is target.
    # Objects: [robot]
    target = np.array(params)
    robot = objects[0]  # robot object
    current_pos = state[robot]

    # Simple P-controller adapted for Grid (Discrete steps)
    # Action: [dx, dy, type]
    diff = target - current_pos[:2]
    dist = np.linalg.norm(diff)

    if dist < 0.1:
        return Action(arr=np.array([0.0, 0.0, 0.0]))

    # Move in the direction of largest difference (Manhattan)
    dx, dy = 0.0, 0.0
    if abs(diff[0]) > abs(diff[1]):
        dx = np.sign(diff[0])
    else:
        dy = np.sign(diff[1])

    return Action(arr=np.array([dx, dy, 0.0]))


def move_initiable(
    state: State, memory: Dict, objects: Sequence[Object], params: Sequence[float]
) -> bool:
    return True  # Always can try to move


def move_terminal(
    state: State, memory: Dict, objects: Sequence[Object], params: Sequence[float]
) -> bool:
    target = np.array(params)
    robot = objects[0]
    current_pos = state[robot]
    dist = np.linalg.norm(target - current_pos[:2])
    return dist < 0.1


def pick_policy(
    state: State, memory: Dict, objects: Sequence[Object], params: Sequence[float]
) -> Action:
    # Action: [vy, vx, grip_cmd]
    # To pick, we just close gripper.
    return Action(arr=np.array([0.0, 0.0, 1.0]))  # 1.0 = close


def pick_initiable(
    state: State, memory: Dict, objects: Sequence[Object], params: Sequence[float]
) -> bool:
    # Robot must be at key location
    # Use Predicate logic or direct state check?
    # Ideally use Predicates if we had them importable easily
    # For now, let's keep direct logic but consistent with domain
    robot, key = objects
    # Or use AtKey.holds(state, objects)?
    return AtKey.holds(state, objects)


def pick_terminal(
    state: State, memory: Dict, objects: Sequence[Object], params: Sequence[float]
) -> bool:
    # Terminal if holding key
    return Holding.holds(state, objects)


def unlock_policy(
    state: State, memory: Dict, objects: Sequence[Object], params: Sequence[float]
) -> Action:
    # Move to door and "interact"?
    # Actually just closing gripper/interacting while at door?
    return Action(arr=np.array([0.0, 0.0, 2.0]))


def unlock_initiable(
    state: State, memory: Dict, objects: Sequence[Object], params: Sequence[float]
) -> bool:
    robot, door, key = objects
    # Or strict signature? Unlock(robot, door)
    # domain logic says AtDoor(robot, door).
    return AtDoor.holds(state, [robot, door])


def unlock_terminal(
    state: State, memory: Dict, objects: Sequence[Object], params: Sequence[float]
) -> bool:
    robot, door, key = objects  # robot, door, key
    return IsOpen.holds(state, [door])


# --- Parameterized Options ---

Move = ParameterizedOption(
    name="Move",
    types=[robot_type],
    policy=move_policy,
    initiable=move_initiable,
    terminal=move_terminal,
)

Pick = ParameterizedOption(
    name="Pick",
    types=[robot_type, key_type],
    policy=pick_policy,
    initiable=pick_initiable,
    terminal=pick_terminal,
)

Unlock = ParameterizedOption(
    name="Unlock",
    types=[robot_type, door_type, key_type],
    policy=unlock_policy,
    initiable=unlock_initiable,
    terminal=unlock_terminal,
)
