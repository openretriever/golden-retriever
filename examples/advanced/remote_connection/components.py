"""
Components for Remote Connection Example.

Defines a Controller (simulated robot interface) and a Policy (simulated heavy compute).
"""

from dataclasses import dataclass
import time
import random

from retriever.flow import Flow, flow_io


@flow_io
@dataclass
class RobotState:
    """Simulated robot state (e.g., joint positions, camera image timestamp)"""
    timestamp: float
    joints: tuple[float, ...]
    image_id: int


@flow_io
@dataclass
class RobotAction:
    """Robot action command"""
    velocity: tuple[float, ...]
    gripper: float


class Controller(Flow[RobotAction, RobotState]):
    """
    Simulated Robot Controller.
    
    - Publishes RobotState at a fixed rate.
    - Receives RobotAction and "executes" it (logs it).
    """
    
    def __init__(self, start_pos: tuple[float, ...] = (0.0, 0.0, 0.0)):
        self.state = list(start_pos)
        self.image_counter = 0

    def step(self, action: RobotAction) -> RobotState:
        # 1. Apply action (simulate movement)
        if action:
            # Simple integration
            dt = 0.1  # Assume 10Hz for simulation
            for i in range(len(self.state)):
                if i < len(action.velocity):
                    self.state[i] += action.velocity[i] * dt
                    
            print(f"[Controller] Executed action: vel={action.velocity}, gri={action.gripper}")
        
        # 2. Simulate sensor reading
        self.image_counter += 1
        
        # 3. Publish state
        return RobotState(
            timestamp=time.time(),
            joints=tuple(self.state),
            image_id=self.image_counter
        )


class Policy(Flow[RobotState, RobotAction]):
    """
    Simulated Compute-Heavy Policy.
    
    - Receives RobotState.
    - Computes action (simulated delay).
    - Publishes RobotAction.
    """
    
    def __init__(self, compute_time: float = 0.05):
        self.compute_time = compute_time

    def step(self, state: RobotState) -> RobotAction:
        print(f"[Policy] Received state: joints={state.joints} (img {state.image_id})")
        
        # Simulate heavy compute (handling image, running neural net)
        time.sleep(self.compute_time)
        
        # Compute dummy action (random drift towards origin)
        action_vel = [-0.1 * p + random.uniform(-0.05, 0.05) for p in state.joints]
        
        return RobotAction(
            velocity=tuple(action_vel),
            gripper=1.0 if state.image_id % 20 < 10 else 0.0
        )
