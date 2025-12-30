
import numpy as np
import time
from dataclasses import dataclass
from typing import Optional

from retriever import Flow
from retriever.flow import flow_io
import pybullet as p
from golden_retriever.envs.ravens.envs.environment import Environment as RavensEnv
from golden_retriever.envs.ravens.tasks.put_block_in_bowl import PutBlockInBowlUnseenColors

from .flows import EnvObservation, EnvAction

class RavensCaPTask(PutBlockInBowlUnseenColors):
    """
    A specialized RAVENS task for Code as Policies that ensures 
    predictable object names and colors for the prompt.
    """
    def __init__(self):
        super().__init__()
        # Override to ensure we get specific colors if possible, 
        # or just rely on the base class and inspect info.

@dataclass
class RavensConfig:
    disp: bool = True  # Enable GUI by default
    hz: int = 240
    n_bowls: int = 1

class RavensEnvFlow(Flow[EnvAction, EnvObservation]):
    def __init__(self, config: RavensConfig = RavensConfig()):
        self.config = config
        self.env = None
        self.task = None
        self.step_count = 0
        self.last_obs = None

    def init(self):
        assets_root = "src/golden_retriever/envs/ravens/envs/assets"  # Assuming run from repo root
        self.env = RavensEnv(assets_root, disp=self.config.disp, hz=self.config.hz)
        self.task = RavensCaPTask()
        self.task.mode = 'train'
        self.env.set_task(self.task)
        self.env.reset()
        print("[RavensEnv] Initialized and reset.")

    def pick_and_place(self, pick_pos, place_pos):
        """
        Execute a pick-and-place action using RAVENS native primitive.
        Args:
            pick_pos: (x, y, z) position to pick from
            place_pos: (x, y, z) position to place at
        Returns:
            success: True if action completed without timeout
        """
        # RAVENS oracle uses identity quaternion (0, 0, 0, 1) for end-effector orientation
        # This means the end-effector points straight down
        pick_rot = (0, 0, 0, 1)  # Identity quaternion - straight down
        place_rot = (0, 0, 0, 1)

        action = {
            'pose0': (tuple(pick_pos), pick_rot),
            'pose1': (tuple(place_pos), place_rot)
        }

        obs, reward, done, info = self.env.step(action)
        return not done  # Return success (True if not timed out)

    def run(self, action: EnvAction) -> EnvObservation:
        """
        Map generic EnvAction to RAVENS primitive actions or just step simulation.
        And return simplified EnvObservation.
        """
        
        # 1. Handle Action
        # RAVENS expects action dict: 'pose0': (pos, rot), 'pose1': (pos, rot)
        # CaP EnvAction: target_pos (x,y,z), gripper_open (bool)
        
        # For this integration, we might need to be stateful to handle 
        # the high-level 'pick' vs 'place' sequences if we want to use RAVENS primitives.
        # OR we can just use the low-level 'move_to' style if we exposed that.
        
        # ACTUALLY, the CaP policy generates `pick(obj)` and `place(loc)`.
        # The `CodePolicyFlow` breaks this down into `EnvAction(target_pos=..., gripper_open=...)`.
        # The `TabletopEnv` (simple impl) just teleported or used simple physics.
        # RAVENS `Environment` has `movej`, `movep`. 
        
        # We need to translate target_pos -> movep.
        
        # Note: RAVENS `step()` assumes a Pick-and-Place primitive if action is provided.
        # If we want continuous control (move_to), we might need to call internal methods 
        # or implement a new "Move" primitive in RAVENS.
        
        # For now, let's keep it simple: 
        # We will use the `env.step_simulation()` for idle ticks.
        # If `action.target_pos` changes, we call `env.movep()`.
        
        if self.env is None:
            return EnvObservation({}, {})

        self.step_count += 1
        
        # Map Action
        if action.target_pos is not None:
             # Convert zlf-retriever coordinates (likely 0-1) to RAVENS coords?
             # RAVENS: Workspace is approx bounds: x=[0.25, 0.75], y=[-0.5, 0.5], z=[0, 0.3]
             # CodePolicyFlow assumes 0-1. We might need remapping or just update prompt to use real coords.
             # Let's assume the LLM uses coordinates provided in context, which come from HERE.
             # So if we report actual RAVENS coords, the LLM will send them back. Consistent.
             
             target = action.target_pos
             # Use movep (blocking in RAVENS standard impl, but we might want it non-blocking?)
             # For this demo, blocking is okay as it runs in its own flow step which might lag.
             # But `movep` takes time. 
             
             # Rudimentary "move to"
             # Orientation: Downwards
             p0 = (target, [0, 1, 0, 0]) 
             self.env.movep(p0, speed=0.01)
        
        if action.gripper_open is not None:
            if action.gripper_open:
                self.env.ee.release()
            else:
                self.env.ee.activate()

        # Step sim a bit to settle
        for _ in range(10): 
            self.env.step_simulation()

        # 2. Build Observation
        # Extract object info for the LLM context
        # env.info map: obj_id -> (pos, rot, dim)
        # We need to map obj_id to semantic names (e.g. "red_block")
        
        # PutBlockInBowlUnseenColors puts colors on objects.
        # We can try to infer or track them.
        
        # Let's inspect `self.env.obj_ids` and cross ref with `self.env.info`.
        
        objects_dict = {}
        for uid in self.env.obj_ids["rigid"]:
            # Basic Hack: Name based on ID
            # In a real system we'd use the Task's knowledge or color.
            name = f"obj_{uid}"
            
            # Query PyBullet for state
            pos, rot = p.getBasePositionAndOrientation(uid)
            
            objects_dict[name] = {
                "position": list(pos),
                "rotation": list(rot)
            }
        


        # Also include fixed objects like bowls?
        for uid in self.env.obj_ids["fixed"]:
             # Filter out plane/workspace/robot base if possible (usually low IDs)
             # But let's verify if they are relevant.
             # Bowls are fixed.
             if uid == self.env.ur5:
                 continue
             
             name = f"fixed_obj_{uid}"
             pos, rot = p.getBasePositionAndOrientation(uid)
             objects_dict[name] = {
                "position": list(pos),
                "rotation": list(rot)
            }
            
        # Get gripper state
        # In RAVENS Suction, check if contact constraint exists.
        holding = None
        if self.env.ee.check_grasp():
            # Find what it is holding
            pass # TODO

        gripper_info = {
            "position": list(p.getLinkState(self.env.ur5, self.env.ee_tip)[0]),
            "open": not self.env.ee.activated, # Suction: active = closed/sucking
            "holding": holding
        }

        return EnvObservation(
            objects=objects_dict,
            gripper=gripper_info,
            time=time.time()
        )

    def finalize(self):
        if self.env:
            p.disconnect()
