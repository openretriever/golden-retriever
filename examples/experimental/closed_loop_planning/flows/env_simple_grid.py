from dataclasses import dataclass
from typing import Tuple

from retriever.flow import Flow

from ..types.flow_types import EnvInput, EnvOutput


@dataclass
class GridState:
    robot_pos: Tuple[int, int] = (0, 0)
    robot_grip: int = 0
    key_loc: Tuple[int, int] = (1, 1)
    door_loc: Tuple[int, int] = (3, 3)
    has_key: bool = False
    door_open: bool = False


class GridEnvironmentFlow(Flow[EnvInput, EnvOutput]):
    def __init__(self, name: str = "GridEnv"):
        self.name = name
        self.state = GridState()
        print(f"[{self.name}] Initialized.")

    def reset(self):
        self.state = GridState()

    def step(self, inp: EnvInput) -> EnvOutput:
        action = inp.action
        cmd_type = 0  # 0: move, 1: pick, 2: unlock
        dx, dy = 0, 0

        if action is not None and action.arr is not None and len(action.arr) >= 3:
            dx = int(action.arr[0])
            dy = int(action.arr[1])
            cmd_type = int(action.arr[2])

        if cmd_type == 0:  # Move
            new_x = max(0, min(5, self.state.robot_pos[0] + dx))
            new_y = max(0, min(5, self.state.robot_pos[1] + dy))
            self.state.robot_pos = (new_x, new_y)
        elif cmd_type == 1:  # Pick
            if self.state.robot_pos == self.state.key_loc:
                self.state.has_key = True
                print(f"[{self.name}] Key Picked!")
        elif cmd_type == 2:  # Unlock
            if self.state.robot_pos == self.state.door_loc and self.state.has_key:
                self.state.door_open = True
                print(f"[{self.name}] Door Unlocked!")

        # Output raw state dict
        obs = {
            "robot": [
                self.state.robot_pos[0],
                self.state.robot_pos[1],
                self.state.robot_grip,
            ],
            "key": [
                self.state.key_loc[0],
                self.state.key_loc[1],
                1.0 if self.state.has_key else 0.0,
            ],
            "door": [
                self.state.door_loc[0],
                self.state.door_loc[1],
                1 if self.state.door_open else 0,
            ],
        }

        return EnvOutput(data=obs)
