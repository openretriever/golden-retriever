
import logging
import dataclasses
import time
import os
import json
from typing import Any, Optional, Deque
from collections import deque
from pathlib import Path

import tyro
import numpy as np
import torch
import cv2
from huggingface_hub import snapshot_download

# Retriever framework
from retriever.flow import Flow, Pipeline, Rate, Trigger

# Local shared imports
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../")
from common.env import OmniGibsonEnv, Observation, Action

# OpenPI Core imports (Model definition only)
try:
    from openpi.policies import policy as _policy
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
except ImportError:
    logging.warning("OpenPI core libraries not found. Creating mocks.")

# -----------------------------------------------------------------------------
# LOGIC INLINED FROM openpi.shared.eval_b1k_wrapper
# -----------------------------------------------------------------------------

RESIZE_SIZE = 224
DEPTH_RESIZE_SIZE = 720

def resize_with_pad(image, target_height, target_width):
    # Simple resize with pad implementation if openpi_client is not available
    h, w = image.shape[:2]
    scale = min(target_height / h, target_width / w)
    nh, nw = int(h * scale), int(w * scale)
    image_resized = cv2.resize(image, (nw, nh))
    
    pad_h = target_height - nh
    pad_w = target_width - nw
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    
    return cv2.copyMakeBorder(image_resized, top, bottom, left, right, cv2.BORDER_CONSTANT)

@dataclasses.dataclass
class Args:
    checkpoint_config: str = "pi05_b1k-base"
    checkpoint: str = "openpi_comet/pi05-b1kpt12-cs32" 
    task_name: str = "turning_on_radio"
    default_prompt: str = "turn on the radio"
    hz: float = 10.0
    
    # Inference parameters
    action_horizon: int = 5
    temporal_ensemble_max: int = 3
    max_len: int = 32

class OpenPIAgent(Flow[Observation, Action]):
    """
    Wraps the OpenPI VLA Policy as a Retriever Flow.
    Inlines logic from B1KPolicyWrapper for full control.
    """
    def __init__(self, args: Args):
        self.args = args
        self.policy = None
        
        # Temporal Ensembling State
        self.action_queue: Deque = deque(maxlen=args.action_horizon)
        self.step_counter = 0
        self.replan_interval = args.action_horizon # K: replan every K steps

    def _ensure_checkpoint(self) -> Path:
        ckpt_path = Path(self.args.checkpoint)
        if ckpt_path.exists():
            return ckpt_path
        logging.info(f"Downloading checkpoint '{self.args.checkpoint}' from HuggingFace...")
        path = snapshot_download(repo_id=self.args.checkpoint)
        return Path(path)

    def _load_policy(self):
        ckpt_path = self._ensure_checkpoint()
        logging.info(f"Loading OpenPI policy from {ckpt_path}...")
        try:
            train_config = _config.get_config(self.args.checkpoint_config)
            self.policy = _policy_config.create_trained_policy(train_config, ckpt_path)
            logging.info("Policy loaded.")
        except Exception as e:
            logging.error(f"Failed to load policy: {e}")

    def _process_obs(self, obs: Observation) -> dict:
        """Process observation to match model input format."""
        
        # 1. Proprioception
        prop_state = obs.proprio[None] # (1, 16)
        
        # 2. Images (Resize and Stack)
        # Expected keys in obs.rgb from OmniGibsonEnv:
        # - robot_r1::robot_r1:zed_link:Camera:0::rgb
        # - robot_r1::robot_r1:left_realsense_link:Camera:0::rgb
        # - robot_r1::robot_r1:right_realsense_link:Camera:0::rgb
        
        rgb_data = obs.rgb
        
        img_obs = np.stack(
            [
                resize_with_pad(rgb_data["robot_r1::robot_r1:zed_link:Camera:0::rgb"], RESIZE_SIZE, RESIZE_SIZE),
                resize_with_pad(rgb_data["robot_r1::robot_r1:left_realsense_link:Camera:0::rgb"], RESIZE_SIZE, RESIZE_SIZE),
                resize_with_pad(rgb_data["robot_r1::robot_r1:right_realsense_link:Camera:0::rgb"], RESIZE_SIZE, RESIZE_SIZE),
            ],
            axis=0 # Stack along first dim? Wrapper did axis=1 but check dimensions: (1, 3, H, W, C)
        )
        # Wrapper: np.stack([img1, img2, img3], axis=1) -> (1, 3, H, W, 3) 
        # But here inputs are (H, W, 3). Stack(axis=0) -> (3, H, W, 3). Add batch -> (1, 3, H, W, 3).
        img_obs = img_obs[None]

        processed = {
            "observation": img_obs,
            "proprio": prop_state
        }
        return processed

    def _act_receeding_temporal(self, input_obs: dict):
        """Standard Temporal Ensembling Strategy."""
        
        # Step 1: Replan if needed
        if self.step_counter % self.replan_interval == 0:
            
            # Format batch for inference
            # nbatch["observation"] is (B, Cam, H, W, C).
            # Model expects dict of specific camera keys.
            
            obs_imgs = input_obs["observation"] # (1, 3, H, W, 3)
            if obs_imgs.shape[-1] != 3:
                 obs_imgs = np.transpose(obs_imgs, (0, 1, 3, 4, 2))
            
            batch = {
                "observation/egocentric_camera": obs_imgs[0, 0],
                "observation/wrist_image_left": obs_imgs[0, 1],
                "observation/wrist_image_right": obs_imgs[0, 2],
                "observation/state": input_obs["proprio"][0],
                "prompt": self.args.default_prompt, # Using fixed prompt
            }

            try:
                # RUN INFERENCE
                action_out = self.policy.infer(batch)
                
                # action_out["actions"] shape: (H, dim) e.g. (32, 16)
                target_joints = action_out["actions"]
                
                # Add to queue
                new_seq = deque([a for a in target_joints[: self.args.max_len]])
                self.action_queue.append(new_seq)
                
            except Exception as e:
                logging.error(f"Inference error: {e}")
                # Don't crash, just rely on existing queue or zeros
                pass

        # Step 2: Smooth / Ensemble
        if not self.action_queue:
            return np.zeros(16) # Fallback

        actions_current_step = np.empty((len(self.action_queue), 16)) # Assuming 16 dim
        
        # Pop one from each sequence in the queue
        valid_indices = []
        for i, q in enumerate(self.action_queue):
            if q:
                actions_current_step[i] = q.popleft()
                valid_indices.append(i)
        
        # Filter empty
        actions_current_step = actions_current_step[valid_indices]
        
        # Exp Weighting (newer actions matter more? or older?)
        # Wrapper logic: exp(k * index) where index 0 is oldest sequence
        k = 0.005
        exp_weights = np.exp(k * np.arange(len(valid_indices)))
        exp_weights = exp_weights / exp_weights.sum()
        
        final_action = (actions_current_step * exp_weights[:, None]).sum(axis=0)
        
        self.step_counter += 1
        return final_action

    def run(self, obs: Observation) -> Action:
        if self.policy is None:
            self._load_policy()
        if self.policy is None:
            return Action(delta_pose=np.zeros(6), gripper=1.0)

        # 1. Process
        input_obs = self._process_obs(obs)
        
        # 2. Act (Temporal Ensemble)
        raw_action = self._act_receeding_temporal(input_obs)
        
        # 3. Parse Action
        # Assuming raw_action is (16,)
        # OpenPI usually: 0-6 pose, -1 gripper.
        return Action(delta_pose=raw_action[:6], gripper=raw_action[-1])


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    
    with Pipeline("behavior_1k_comet") as pipe:
        # 1. Environment
        env = OmniGibsonEnv(task_name=args.task_name) @ Rate(hz=args.hz)
        
        # 2. Agent
        agent = OpenPIAgent(args) @ Trigger("observation")

        # 3. Connect
        env >> agent >> env

    logging.info("Starting Retriever pipeline...")
    pipe.reset()
    
    try:
        while True:
            start_time = time.time()
            pipe.step(dt=1.0/args.hz)
            
            elapsed = time.time() - start_time
            sleep_time = max(0, (1.0/args.hz) - elapsed)
            time.sleep(sleep_time)
            
    except KeyboardInterrupt:
        logging.info("Simulation stopped by user.")

if __name__ == "__main__":
    main(tyro.cli(Args))
