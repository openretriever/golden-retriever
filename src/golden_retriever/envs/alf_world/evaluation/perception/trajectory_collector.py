# This script is for collecting trajectories in ALFWorld.

import argparse
import json
import os
import random

import numpy as np
import torch
from agents.agent_utils.planner import Planner, ReActPlanner
from PIL import Image
from utils import AlfEnv, load_config_file

os.environ["ALFWORLD_DATA"] = "./data_storage"


def save_tensor_as_png(tensor: torch.Tensor, save_path: str):
    """
    Save a tensor in the form of [H, W, C] as a PNG image.

    Args:
        tensor (torch.Tensor): Input tensor in the form of [H, W, C] with values in [0,255]
        save_path (str): Path where to save the PNG file
    """
    # Ensure tensor is on CPU and convert to numpy
    tensor = tensor.cpu().numpy().astype(np.uint8)

    # Convert to PIL Image
    image = Image.fromarray(tensor)

    # Save as PNG
    image.save(save_path, format="PNG")


def main():
    # set random seed for reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_traj", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--agent",
        type=str,
        choices=["react", "rule_based_expert"],
        default="rule_based_expert",
        help="Type of agent to use for trajectory collection",
    )
    args = parser.parse_args()

    num_traj = args.num_traj
    horizon = args.horizon
    seed = args.seed

    random.seed(seed)
    np.random.seed(seed)

    if args.agent == "react":
        agent = ReActPlanner(llm_model="gpt-4o")
    elif args.agent == "rule_based_expert":
        agent = Planner()

    # loading envs
    config_file = "./alf-config.yaml"
    config = load_config_file(config_file)
    assert "AlfredThorEnv" in config["env"]["type"], "Only AlfredThorEnv is supported"
    envs = AlfEnv(config_file, train_eval="train", obs_type="both")
    obs, infos = envs.reset(seed=seed)
    # Check if data/trajectory directory exists
    save_dir = f"./data/{args.agent}/trajectory"
    if not os.path.exists(save_dir):
        print(
            f"Warning: Directory {save_dir} does not exist. Skipping trajectory save."
        )
        return

    ## Collect trajectories
    trajectories = []  # List to store all trajectories

    for traj_idx in range(num_traj):
        # Using name to save the trajectory

        name = "/".join(infos["extra.gamefile"][0].split("/")[-2:])
        start_info = "\n".join(infos["observation_text"][0].split("\n\n")[1:])
        agent.get_task_info(name, start_info)
        # Initialize trajectory data structure
        current_trajectory = {
            "game_file": name,
            "initial_info": start_info,
            "steps": [],
            "won": False,
        }
        task_dir = os.path.join(save_dir, name)
        os.makedirs(task_dir, exist_ok=True)
        segs_dir = os.path.join(task_dir, "segs")
        os.makedirs(segs_dir, exist_ok=True)
        bbox_dir = os.path.join(task_dir, "bbox")
        os.makedirs(bbox_dir, exist_ok=True)
        for steps in range(horizon):
            # action = infos["extra.expert_plan"][0][0]
            action = (
                agent.predict()
                if args.agent == "react"
                else infos["extra.expert_plan"][0][0]
            )
            cur_obs, reward, done, infos = envs.step([action])
            agent.add_to_memory("observation", infos["observation_text"][0])
            save_tensor_as_png(
                cur_obs["segs"]["image"], os.path.join(segs_dir, f"{steps}.png")
            )
            save_tensor_as_png(
                cur_obs["bbox"]["image"], os.path.join(bbox_dir, f"{steps}.png")
            )
            # Log step information
            step_info = {
                "step": steps,
                "action": action,
                "observation": infos["observation_text"][0],
                "object_bindings": cur_obs["bbox"]["object_bindings"],
                "frame_description": infos["frame_description"],
                "inventory": infos["inventory"],
            }
            current_trajectory["steps"].append(step_info)
            print(
                f"> action: {action}\n> obs: {infos['observation_text'][0]} \n> frame_description: {infos['frame_description']}"
            )

            if done[0] or steps == horizon - 1:
                current_trajectory["won"] = "True" if infos["won"][0] else "False"
                trajectories.append(current_trajectory)
                # Save individual trajectory to a separate file
                # Create subdirectory based on task name

                traj_filename = "traj.json"
                traj_path = os.path.join(task_dir, traj_filename)
                with open(traj_path, "w") as f:
                    json.dump(current_trajectory, f, indent=2)
                print(f"Saved trajectory {traj_idx} to {traj_path}")
                obs, infos = envs.reset(seed=seed)
                agent.reset()
                break


if __name__ == "__main__":
    main()
