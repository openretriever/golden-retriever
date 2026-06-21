"""
Flow wrappers for Code as Policies.
"""

from dataclasses import dataclass, field
from typing import Optional, Any
import logging
import queue
import time
import numpy as np

from retriever import Flow
from retriever.flow import io
from .env import TabletopEnv
from .executor import PolicyExecutor, ExecutionRequest
from .agent import CodeGenAgent

logger = logging.getLogger(__name__)

# --- I/O Definitions ---

@io
@dataclass
class EnvObservation:
    objects: dict
    gripper: dict
    time: float = 0.0

@io
@dataclass
class EnvAction:
    target_pos: Optional[list[float]] = None
    gripper_open: Optional[bool] = None

@io
@dataclass
class UserInstruction:
    text: str

# --- Flows ---

class TabletopEnvFlow(Flow[EnvAction, EnvObservation]):
    """
    Wraps TabletopEnv.
    """
    def __init__(self):
        super().__init__()
        # Initial scene setup
        self.init_objects = [
            {"name": "red_block", "position": [0.3, 0.3], "color": "red"},
            {"name": "blue_block", "position": [0.7, 0.7], "color": "blue"},
            {"name": "bowl", "position": [0.5, 0.8], "color": "green"},
        ]
    
    def init(self):
        self.env = TabletopEnv(self.init_objects)
        self._obs = self.env.reset()
        logger.info("Env initialized.")

    def run(self, action: EnvAction) -> EnvObservation:
        # Convert Flow Action to Env Dict
        act_dict = {}
        if action.target_pos is not None:
            act_dict["target_pos"] = action.target_pos
        if action.gripper_open is not None:
            act_dict["gripper_open"] = action.gripper_open
            
        obs, _, _, _ = self.env.step(act_dict)
        self._obs = obs
        
        return EnvObservation(
            objects=obs["objects"],
            gripper=obs["gripper"],
            time=time.time()
        )

class CodePolicyFlow(Flow[EnvObservation, EnvAction]):
    """
    Manages the User Instruction -> LLM -> Code Execution pipeline.
    Also acts as the 'bridge' for the threaded executor.
    """

    def __init__(self, instruction: str, model: str = None, env_flow=None):
        super().__init__()
        self.instruction = instruction
        self.model = model
        self.env_flow = env_flow  # Optional handle for environment-specific helpers.
        self.agent = None
        self.executor = None
        self.generated_code = None
        self.execution_started = False

        # State tracking for pick-place
        self.pending_pick_pos = None  # Store pick position for combined action
        self.current_req: Optional[ExecutionRequest] = None

    def init(self):
        self.agent = CodeGenAgent(model=self.model)
        self.executor = PolicyExecutor()
        logger.info(f"Policy initialized with task: {self.instruction}")

    def run(self, obs: EnvObservation) -> EnvAction:
        # 1. Trigger Generation on first step (or if we had a dynamic trigger)
        if not self.generated_code:
            logger.info("Generating code...")
            obj_names = list(obs.objects.keys())
            self.generated_code = self.agent.generate_code(self.instruction, obj_names)
            logger.info(f"Generated Code:\n{self.generated_code}")
            
            # Start Execution
            self.executor.start_execution(self.generated_code)
            self.execution_started = True

        # 2. Process Executor Queue
        # Check if there is a pending request from the thread
        if self.current_req is None:
            try:
                self.current_req = self.executor.request_queue.get_nowait()
                logger.info(f"[Policy] Received request: {self.current_req.command}")
            except queue.Empty:
                pass
        
        # 3. Handle Active Request
        action = EnvAction() # Default no-op
        
        if self.current_req:
            req = self.current_req
            if req.command == "say":
                logger.info(f"[ROBOT SAYS]: {req.args[0]}")
                self._complete_req(None)
                
            elif req.command == "get_object_position":
                name = req.args[0]
                if name in obs.objects:
                    pos = obs.objects[name]["position"]
                    self._complete_req(pos)
                else:
                    self._complete_req(None, error=ValueError(f"Object {name} not found"))

            elif req.command == "move_to":
                target = req.args[0] # (x, y)
                current_gripper = np.array(obs.gripper["position"])
                target_3d = [target[0], target[1], current_gripper[2]] # Maintain Z
                
                # Check completion (simple distance threshold)
                dist = np.linalg.norm(np.array(target_3d[:2]) - current_gripper[:2])
                if dist < 0.05:
                    self._complete_req(None)
                else:
                    # Output action to move
                    action.target_pos = target_3d
            
            elif req.command == "pick":
                self._handle_pick_macro(req, obs, action)

            elif req.command == "place":
                self._handle_place_macro(req, obs, action)

        return action

    def _complete_req(self, result: Any, error: Exception = None):
        """Mark current request as done and notify thread."""
        if self.current_req:
            self.current_req.result = result
            self.current_req.error = error
            self.current_req.event.set()
            self.current_req = None

    # --- Macros ---
    # These store state in specific variables to progress over multiple ticks
    
    def _handle_pick_macro(self, req, obs, action):
        # Extremely simplified sequence for demo
        # 1. Get obj pos
        # 2. Move to top
        # 3. Lower (Z=0.05)
        # 4. Close
        # 5. Lift (Z=0.5)
        
        # We use a static counter on the request object for simplicity (hacky but works for demo)
        if not hasattr(req, "_state"):
            req._state = 0
            
        obj_name = req.args[0]
        if obj_name not in obs.objects:
            self._complete_req(None, ValueError(f"Unknown object {obj_name}"))
            return

        obj_pos = obs.objects[obj_name]["position"]
        gripper_pos = np.array(obs.gripper["position"])
        
        # State 0: Move XY
        if req._state == 0:
            target = [obj_pos[0], obj_pos[1], 0.20]  # Approach height within workspace
            if np.linalg.norm(gripper_pos[:2] - np.array(target[:2])) < 0.05:
                req._state = 1
            else:
                action.target_pos = target
                action.gripper_open = True
        
        # State 1: Lower
        elif req._state == 1:
            target = [obj_pos[0], obj_pos[1], 0.02]  # Grasp height
            if np.abs(gripper_pos[2] - target[2]) < 0.05:
                req._state = 2
            else:
                action.target_pos = target
                action.gripper_open = True

        # State 2: Close
        elif req._state == 2:
            if obs.gripper["holding"] == obj_name or not obs.gripper["open"]:
                req._state = 3
            else:
                action.target_pos = gripper_pos.tolist()
                action.gripper_open = False
        
        # State 3: Lift
        elif req._state == 3:
            target = [gripper_pos[0], gripper_pos[1], 0.20]  # Lift within workspace
            if np.abs(gripper_pos[2] - target[2]) < 0.05:
                # Done
                self._complete_req(None)
            else:
                action.target_pos = target
                action.gripper_open = False

    def _handle_place_macro(self, req, obs, action):
        if not hasattr(req, "_state"):
            req._state = 0
            
        target_loc = req.args[0] # (x, y)
        gripper_pos = np.array(obs.gripper["position"])
        
        # State 0: Move XY
        if req._state == 0:
            target = [target_loc[0], target_loc[1], 0.20]  # Approach height
            if np.linalg.norm(gripper_pos[:2] - np.array(target[:2])) < 0.05:
                req._state = 1
            else:
                action.target_pos = target
                action.gripper_open = False # Keep holding
        
        # State 1: Lower
        elif req._state == 1:
            target = [target_loc[0], target_loc[1], 0.10]  # Release height
            if np.abs(gripper_pos[2] - target[2]) < 0.05:
                req._state = 2
            else:
                action.target_pos = target
                action.gripper_open = False

        # State 2: Open
        elif req._state == 2:
            if obs.gripper["open"]:
                req._state = 3
            else:
                action.target_pos = gripper_pos.tolist()
                action.gripper_open = True
        
        # State 3: Lift
        elif req._state == 3:
            target = [gripper_pos[0], gripper_pos[1], 0.20]  # Retreat height
            if np.abs(gripper_pos[2] - target[2]) < 0.05:
                # Done
                self._complete_req(None)
            else:
                action.target_pos = target
                action.gripper_open = True

    def finalize(self):
        if self.executor:
            self.executor.stop()
