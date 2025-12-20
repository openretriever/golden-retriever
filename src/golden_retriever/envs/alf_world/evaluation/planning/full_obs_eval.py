"""
This script is used to evaluate the agent's performance in the environment with full observability.
"""

import argparse

from agents.agent_utils.planner import ReActPlanner
from utils import AlfEnv, load_config_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-eval", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=30)
    args = parser.parse_args()

    config_file = "./alf-config.yaml"
    config = load_config_file(config_file)
    assert "AlfredThorEnv" in config["env"]["type"], "Only AlfredThorEnv is supported"
    envs = AlfEnv(config_file, train_eval="eval_out_of_distribution", obs_type="both")
    planner = ReActPlanner(llm_model="gpt-4o")

    obs, infos = envs.reset()

    for _ in range(args.num_eval):
        name = "/".join(infos["extra.gamefile"][0].split("/")[-2:])
        start_info = "\n".join(infos["observation_text"][0].split("\n\n")[1:])
        task_objects = envs.get_task_objects()
        location_infos = [
            envs.encode_object_locations(task_object) for task_object in task_objects
        ]
        # append the infos that how many task_objects are in the receptacles to start_info before "Your task is to..."
        parts = start_info.split("Your task is to")
        start_info = (
            f"{parts[0]}{', '.join(location_infos)}.\nYour task is to{parts[1]}"
        )
        planner.get_task_info(task_name=name, task_info=start_info)

        for step in range(args.horizon):
            action = planner.predict()
            _, _, _, infos = envs.step([action])
            planner.add_to_memory("observation", infos["frame_description"])
            print(
                f"> action: {action}\n> obs: {infos['observation_text'][0]} \n> frame_description: {infos['frame_description']}"
            )
            if step == args.horizon - 1 or infos["won"][0]:
                planner.reset()
                _, infos = envs.reset(seed=args.seed)
                break


if __name__ == "__main__":
    main()
