"""
Simple 2D Tabletop Environment for Code as Policies.
"""

from dataclasses import dataclass, field
import numpy as np
import logging

logger = logging.getLogger(__name__)

@dataclass
class ObjectState:
    name: str
    position: np.ndarray  # [x, y]
    color: str
    held: bool = False

@dataclass
class GripperState:
    position: np.ndarray  # [x, y, z] (z=0 is table, z=1 is safe height)
    open: bool = True
    holding: str | None = None

class TabletopEnv:
    """
    A simple 2D simulation of a robot arm moving on a tabletop.
    """

    def __init__(self, objects: list[dict]):
        self.gripper = GripperState(position=np.array([0.5, 0.5, 0.5]), open=True)
        self.objects = {}
        for obj in objects:
            self.objects[obj["name"]] = ObjectState(
                name=obj["name"],
                position=np.array(obj["position"], dtype=np.float32),
                color=obj.get("color", "gray")
            )
        self.dt = 0.1  # Simulation time step

    def reset(self):
        # Reset could randomize positions, but for now we keep init
        self.gripper = GripperState(position=np.array([0.5, 0.5, 0.5]), open=True)
        for obj in self.objects.values():
            obj.held = False
            # Ideally reset positions to initial if we stored them
        return self._get_obs()

    def step(self, action: dict):
        """
        Execute a low-level action.
        Action dict:
        - "target_pos": [x, y, z] (optional)
        - "gripper_open": bool (optional)
        """
        done = False
        info = {}

        # 1. Update Gripper Position (Instant/Teleport for simplicity, or interpolated)
        if "target_pos" in action:
            # Simple P-control or teleport
            # Here we do a simple interpolation step limit
            target = np.array(action["target_pos"], dtype=np.float32)
            dist = np.linalg.norm(target - self.gripper.position)
            max_step = 0.1
            if dist > max_step:
                direction = (target - self.gripper.position) / dist
                self.gripper.position += direction * max_step
            else:
                self.gripper.position = target

        # 2. Update Gripper State
        if "gripper_open" in action:
            was_open = self.gripper.open
            self.gripper.open = action["gripper_open"]
            
            # Grasp logic
            if was_open and not self.gripper.open:
                # Try to grasp
                if self.gripper.holding is None:
                    # Find closest object
                    for name, obj in self.objects.items():
                        if not obj.held:
                            # 2D distance ignoring Z for grasp check, but Z must be low
                            d = np.linalg.norm(obj.position - self.gripper.position[:2])
                            if d < 0.1 and self.gripper.position[2] < 0.1:
                                self.gripper.holding = name
                                obj.held = True
                                logger.info(f"[Env] Grasping {name}")
                                break
            
            # Release logic
            elif not was_open and self.gripper.open:
                if self.gripper.holding:
                    logger.info(f"[Env] Releasing {self.gripper.holding}")
                    self.objects[self.gripper.holding].held = False
                    self.gripper.holding = None

        # 3. Update Held Object Position
        if self.gripper.holding:
            obj = self.objects[self.gripper.holding]
            # Object follows gripper
            obj.position = self.gripper.position[:2].copy()

        return self._get_obs(), 0.0, done, info

    def _get_obs(self):
        # Return simple dict representation
        return {
            "gripper": {
                "position": self.gripper.position.tolist(),
                "open": self.gripper.open,
                "holding": self.gripper.holding
            },
            "objects": {
                name: {
                    "position": obj.position.tolist(),
                    "held": obj.held,
                    "color": obj.color
                }
                for name, obj in self.objects.items()
            }
        }
