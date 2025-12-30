from typing import Set

import numpy as np

from retriever.flow import Flow
from retriever.types.symbolic import GroundAtom, Object, State

from ..types.domain import (
    AtDoor,
    AtKey,
    Holding,
    IsOpen,
    door_type,
    key_type,
    robot_type,
)
from ..types.flow_types import PerceptionInput, PerceptionOutput


class PerceptionFlow(Flow[PerceptionInput, PerceptionOutput]):
    def __init__(self, name: str = "PerceptionFlow"):
        self.name = name
        # Rerun initialized globally

    def step(self, input: PerceptionInput) -> PerceptionOutput:
        import rerun as rr
        obs = input.data
        if not obs:
            print(f"[{self.name}] Warning: Empty observation received.")
            rr.log("perception/status", rr.TextDocument("Empty Observation"))
            return PerceptionOutput(state=None, atoms=set())

        print(f"[{self.name}] Processing observation: Robot={obs.get('robot')}")

        # 1. Parse Objects from Observation
        robot_obs = obs.get("robot", [0, 0, 0])
        key_obs = obs.get("key", [5, 5, 0])
        door_obs = obs.get("door", [8, 8, 0])

        # Instantiate Typed Objects
        robot_obj = Object("robot", robot_type)
        key_obj = Object("key", key_type)
        door_obj = Object("door", door_type)

        # Create State
        state_data = {
            robot_obj: np.array(robot_obs),
            key_obj: np.array(key_obs),
            door_obj: np.array(door_obs),
        }

        # Add Global Image if present (for VLM)
        if "image" in obs:
            state_data["global_image"] = obs["image"]

        state = State(data=state_data)

        # 2. Compute Ground Atoms (Simulated Perception)
        atoms: Set[GroundAtom] = set()

        if AtKey.holds(state, [robot_obj, key_obj]):
            atoms.add(GroundAtom(AtKey, [robot_obj, key_obj]))

        if AtDoor.holds(state, [robot_obj, door_obj]):
            atoms.add(GroundAtom(AtDoor, [robot_obj, door_obj]))

        if Holding.holds(state, [robot_obj, key_obj]):
            atoms.add(GroundAtom(Holding, [robot_obj, key_obj]))

        if IsOpen.holds(state, [door_obj]):
            atoms.add(GroundAtom(IsOpen, [door_obj]))

        # Log atoms to Rerun
        if atoms:
            atom_text = "\n".join([str(a) for a in atoms])
            rr.log("perception/atoms", rr.TextDocument(atom_text))
        else:
            rr.log("perception/atoms", rr.TextDocument("None"))

        return PerceptionOutput(state=state, atoms=atoms)
