
import logging
import time
from dataclasses import dataclass
import numpy as np

from retriever.flow import Flow, flow_io
from examples.advanced.hierarchical_vla.perception import GoalEmbedding

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

logger = logging.getLogger(__name__)

@flow_io
@dataclass
class RobotState:
    joint_angles: np.ndarray
    timestamp: float

@flow_io
@dataclass
class RobotAction:
    motor_torques: np.ndarray
    timestamp: float

class ControlPolicy(nn.Module):
    def __init__(self, state_dim=5, goal_dim=10, action_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + goal_dim, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim),
            nn.Tanh()
        )
        
    def forward(self, state, goal):
        x = torch.cat([state, goal], dim=-1)
        return self.net(x)

class ControlFlow(Flow[RobotState, RobotAction]):
    """
    Simulates a "Fast" Control loop ( ~50 Hz ).
    
    It consumes the *latest available* GoalEmbedding from Perception.
    It does NOT block if Perception is slow.
    """
    def __init__(self):
        self.policy = None
        self.current_goal = None
        
    def init(self):
        if torch is None:
            raise ImportError("PyTorch required for ControlFlow")
            
        self.policy = ControlPolicy()
        self.policy.eval()
        # Default goal: zeros
        self.current_goal = torch.zeros(10)
        logger.info("[Control] Policy initialized.")

    def run(self, robot_state: RobotState) -> RobotAction:
        """
        Note: In a real Retriever setup, we would often combine 
        (RobotState, GoalEmbedding) into a single input tuple/dataclass.
        
        However, to demonstrate 'Async Latest', we will handle the goal
        injection via a separate async mechanism or just assume we have access 
        to it. 
        
        Wait! Standard Flow `run` takes ONE input type.
        If we want to combine streams (State + Goal), we usually use a
        Pipeline `join` or `combine`. 
        
        For this example, let's assume the Input to this Flow IS the combined state
        provided by the Runtime, OR we manually handle the goal update if we act as a sink.
        
        Design Choice:
        Let's make the input a `ControlInput` which contains both.
        The `app.py` will configure the pipeline such that `Goal` is sampled LATEST
        and `State` is sampled immediate/synced.
        """
        # ERROR in logic above: `run` signature is fixed by Flow[I, O].
        # I need to define a merged input type.
        pass

# Redefining with correct input
    
@flow_io
@dataclass
class ControlInput:
    # From RobotState
    joint_angles: np.ndarray
    state_timestamp: float
    # From GoalEmbedding
    goal_vector: np.ndarray
    goal_timestamp: float

class ControlFlow(Flow[ControlInput, RobotAction]):
    def __init__(self):
        self.policy = None
        self._step_count = 0
        
    def init(self):
        if torch is None: raise ImportError("PyTorch")
        self.policy = ControlPolicy()
        self.policy.eval()
        logger.info("[Control] Policy initialized.")

    def run(self, inputs: ControlInput) -> RobotAction:
        self._step_count += 1
        
        # Handle startup: inputs may be None before first data arrives
        if inputs.joint_angles is None:
            # Return a no-op action
            return RobotAction(
                motor_torques=np.zeros(5, dtype=np.float32),
                timestamp=time.time()
            )
        
        # Convert to torch
        s = torch.from_numpy(inputs.joint_angles).float()
        
        # If goal is missing (startup/zeros), use zeros
        # Note: In flattened flow, if goal_vector is None? 
        # Actually with @flow_io, fields are required usually.
        # But if we use 'Latest', it might wait for first value?
        # If goal_vector comes from `perception`, and valid.
        
        if inputs.goal_vector is not None:
            g = torch.from_numpy(inputs.goal_vector).float()
        else:
            g = torch.zeros(10)
            
        with torch.no_grad():
            action_tensor = self.policy(s, g)
            
        if self._step_count % 50 == 0:
            logger.info(f"[Control] Step {self._step_count}: Action generated (goal valid: {inputs.goal_vector is not None})")

        return RobotAction(
            motor_torques=action_tensor.numpy(),
            timestamp=time.time()
        )
