from dataclasses import dataclass
from typing import Any, Dict, Tuple

from rich import print as rprint
from rich.panel import Panel

from retriever.flow import Flow, flow_io

from ..types.flow_types import EnvInput, EnvOutput


@dataclass
class SpotState:
    robot_pose: Tuple[float, float, float] = (0.0, 0.0, 0.0) # x, y, yaw
    battery_level: float = 100.0
    status: str = "IDLE"

@flow_io
@dataclass
class RiseEnvOutput(EnvOutput):
    """Output from the RISE environment."""
    data: dict = None  # Match EnvOutput/PerceptionInput type

class RiseEnvironmentFlow(Flow[EnvInput, RiseEnvOutput]):
    """RISE (Real-Image Synthetic Environment) Flow.
    
    Simulates Spot robot behavior using synthetic or pre-recorded data,
    structured identically to the real Spot environment.
    """
    def __init__(self, name: str = "RiseEnvironmentFlow"):
        self.name = name
        self.state = SpotState()

        # Rerun is initialized by DoraExecutor automatically
        rprint(Panel(f"[{self.name}] Initialized. Rerun active.", title="RISE System"))

    def reset(self):
        self.state = SpotState()
        rprint(f"[{self.name}] Reset.")

    def step(self, inp: EnvInput) -> RiseEnvOutput:
        import rerun as rr
        # Log Logic Cycle
        rr.set_time_seconds("sim_time", self.state.robot_pose[0]) # Mock time advancement?

        if inp.action:
            if inp.action.arr is not None:
                dx, dy = inp.action.arr[0], inp.action.arr[1]
                rprint(f"[{self.name}] [bold green]Action[/]: [{dx:.2f}, {dy:.2f}] | Pose: {self.state.robot_pose}")

                # Update state
                x, y, yaw = self.state.robot_pose
                self.state.robot_pose = (x + dx, y + dy, yaw)
            else:
                 rprint(f"[{self.name}] [bold yellow]Action (No Array)[/]: {inp.action}")

        # Mock Image for VLM testing
        # Create a simple red/green image based on door state
        from PIL import Image
        import numpy as np
        
        # Simple traffic light image: Green if door open (mocking logical state), Red otherwise?
        # For now, just fixed image
        img_array = np.zeros((100, 100, 3), dtype=np.uint8)
        img_array[:, :, 1] = 255 # Green
        mock_image = Image.fromarray(img_array)

        # Output mock data
        obs: Dict[str, Any] = {
            "robot": list(self.state.robot_pose),
            "key": [5, 5, 0],
            "door": [8, 8, 0],
            "battery": self.state.battery_level,
            "image": mock_image # Add mock image for VLM
        }

        # Log to Rerun
        rr.log("world/robot", rr.Points3D([list(self.state.robot_pose)], colors=[[255, 0, 0]], radii=[0.2], labels=["Spot"]))
        rr.log("world/key", rr.Points3D([[5, 5, 0]], colors=[[255, 255, 0]], radii=[0.1], labels=["Key"]))
        rr.log("world/door", rr.Points3D([[8, 8, 0]], colors=[[0, 255, 0]], radii=[0.3], labels=["Door"]))

        return RiseEnvOutput(data=obs)
