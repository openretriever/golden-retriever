from typing import Any, Dict, List, Optional
import time

from retriever.flow import Flow, flow_io
from dataclasses import dataclass, field
from retriever.types.options import Action

# Using Any for types to avoid complex dependency chains in this example
@flow_io
@dataclass
class EnvInput:
    action: Optional[Action] = None 

@flow_io
@dataclass
class EnvOutput:
    observation: Any = None
    camera_images: dict = field(default_factory=dict)
    # Joint states etc
    joint_state: Dict[str, float] = field(default_factory=dict)

class ManipulationEnvFlow(Flow[EnvInput, EnvOutput]):
    """
    Simulated Manipulation Environment.
    Provides camera feeds and joint states, distinct from the controller.
    """
    def __init__(self, name: str = "ManipulationEnv"):
        self.name = name
        self.steps = 0

    def step(self, inp: EnvInput) -> EnvOutput:
        self.steps += 1
        # Mock simulation logic
        
        # Simulate camera images (placeholder)
        images = {
            "hand_camera": f"image_hand_{self.steps}",
            "fixed_camera": f"image_fixed_{self.steps}"
        }
        
        # Simulate joint state
        joints = {
            "joint1": 0.1 * self.steps,
            "joint2": -0.1 * self.steps
        }
        
        return EnvOutput(
            camera_images=images,
            joint_state=joints
        )
