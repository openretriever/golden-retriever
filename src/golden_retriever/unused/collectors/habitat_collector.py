""" This script is used to collect offline dataset.
"""
import os

import numpy as np

from src.mappers.utils_vlmaps.utils.parser import YamlParser
from src.models.mapping.utils_vlmaps.utils import create_offline_dataset_dirs
from src.utils.simulator_utils import make_env

# Quiet the Habitat log
os.environ["HABITAT_SIM_LOG"] = "quiet"


class CollectDataset:
    """
    Manually collect one episode.
        - w: move_forward
        - a: turn_left
        - d: turn_right
        - q: exit
    """

    def __init__(self, config, sim_config):
        # General configuration
        self.run_cfg = config
        self.sim_cfg = sim_config

        # Simulator
        self.simulator = None

        # Define the manual actions
        self.valid_actions = {
            "w": "move_forward",
            "a": "turn_left",
            "d": "turn_right",
            "q": "exit",
        }

    @staticmethod
    def save_offline_data(observation, time_step, save_dir_path):
        # Split the observation: RGB image, Depth image, and Pose (position + quaternion)
        rgb_img = observation["color_sensor"][:, :, 0:3]
        depth_img = observation["depth_sensor"]
        pose_info = observation["position"] + observation["quaternion"]
        # Save the data
        np.save(f"{save_dir_path}/RGB/rgb_img_{time_step}.npy", rgb_img)
        np.save(
            f"{save_dir_path}/Depth/depth_img_{time_step}.npy",
            depth_img,
        )
        np.save(f"{save_dir_path}/Pose/pose_{time_step}.npy", pose_info)

    def keyboard_input_action(self, t):
        # Input the action from the keyboard
        while True:
            # Input the action
            action = input(f"Time step {t} action = ")
            # Check action validation
            if action in self.valid_actions.keys():
                return self.valid_actions[action]
            else:
                print("Invalid action. Please input again.")
                print("-----------------------------------")

    def close(self):
        self.simulator.close()

    def get_action(self, time_step: int) -> str:
        """
        We provide three ways to generate the next action.
        Set the self.run_cfg['mode'] to set the action generation.
        'manual': keyboard input
        'auto': optimal planner
        'random': random action
        """
        if self.run_cfg["mode"] == "manual":
            action = self.keyboard_input_action(time_step)  # input from keyboard
            action = "exit" if action == "q" else action
        elif self.run_cfg["mode"] == "auto":
            action = self.simulator.optimal_actions[time_step]  # use the optimal action
            action = "exit" if action == "stop" else action
        elif self.run_cfg["mode"] == "random":
            action = np.random.choice(self.simulator.action_space, 1)[
                0
            ]  # sample one random action
            action = "exit" if action == "stop" else action
        else:
            action = "exit"

        return action

    def run(self):
        # create simulator
        self.sim_cfg["scene_cfg"]["scene_file"] = (
            self.run_cfg["dataset_dir_path"] + self.run_cfg["scene_id"]
        )
        self.simulator = make_env(self.sim_cfg)
        self.simulator.reset()

        for epi_idx in range(self.run_cfg["episode_num"]):
            # create saving directories for one episode
            save_dir_path = (
                f"{self.run_cfg['save_dir_path']}/"
                f"{self.run_cfg['mode']}/"
                f"{run_config['scene_id'].split('.')[0]}/"
                f"episode_{epi_idx}"
            )
            create_offline_dataset_dirs(save_dir_path)

            # reset the episode
            observation = self.simulator.reset()

            # render observations for visualization
            self.simulator.render(observation, time_step=epi_idx)

            # save the first observations
            self.save_offline_data(
                observation, time_step=0, save_dir_path=save_dir_path
            )

            # collect one episode
            for t in range(1, self.sim_cfg["scene_cfg"]["max_episode_length"]):
                # Manually input one action
                action = self.get_action(time_step=t)

                # break when exit
                if action == "exit":
                    break

                # step
                observation, done, truncated = self.simulator.step(action)

                # save the data
                self.save_offline_data(
                    observation, time_step=t, save_dir_path=save_dir_path
                )

                # render observation
                self.simulator.render(observation, time_step=t)

                # determinate the episode if the goal is reached
                if done or truncated:
                    break

        # close the environment
        self.close()


# Main to run
if __name__ == "__main__":
    # load simulation configurations
    simulator_config = YamlParser("config/habitat_config.yaml").data

    # set the scene path, scene name, and save path
    run_config = {
        "dataset_dir_path": "src/env/gibson_example/",
        "scene_id": "Eudora.glb",
        "save_dir_path": "data",
        "episode_num": 3,
        "mode": "manual",
    }

    # collect offline dataset
    collector = CollectDataset(run_config, simulator_config)
    collector.run()
