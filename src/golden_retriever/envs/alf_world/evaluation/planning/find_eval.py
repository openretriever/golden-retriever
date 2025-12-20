"""
This script is used to evaluate if the agent can find the object in the environment. If found, how many steps it takes.
"""

import argparse
import random
import re

import numpy as np
from agents.agent_utils.planner import ReActPlanner
from utils import AlfEnv, load_config_file


def transform_task(description, task_object):
    # Match the task line and extract the objects to find
    task_match = re.search(r"Your task is to: (.+)", description)
    if task_match:
        # replace the task line to find the object
        return description.replace(
            task_match.group(1), f"find a {task_object} and take it."
        )
    return description


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    # set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)

    # loading envs
    config_file = "./alf-config.yaml"
    config = load_config_file(config_file)
    assert "AlfredThorEnv" in config["env"]["type"], "Only AlfredThorEnv is supported"
    envs = AlfEnv(config_file, train_eval="train", obs_type="both")
    obs, infos = envs.reset(seed=args.seed)

    num_eval = args.num_eval
    horizon = args.horizon

    planner = ReActPlanner(llm_model="gpt-4o")

    for _ in range(num_eval):
        # rewrite the start info to finding the object
        start_info = "\n".join(infos["observation_text"][0].split("\n\n")[1:])
        task_object = envs.env.envs[0].traj_data["pddl_params"]["object_target"].lower()
        start_info = transform_task(start_info, task_object)
        planner.get_task_info("find_object", start_info)
        for steps in range(horizon):
            # action = infos["extra.expert_plan"][0][0]
            action = planner.predict()
            _, _, _, infos = envs.step([action])
            planner.add_to_memory("observation", infos["frame_description"])
            print(
                f"> action: {action}\n> obs: {infos['observation_text'][0]} \n> frame_description: {infos['frame_description']}"
            )

            if task_object in infos["inventory"] or steps == horizon - 1:
                planner.reset()
                _, infos = envs.reset(seed=args.seed)
                break


if __name__ == "__main__":
    main()
