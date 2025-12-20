"""Data collection script."""

import os
import pathlib
import random

import hydra
import numpy as np

from retriever.envs.ravens import tasks
from retriever.envs.ravens.dataset import RavensDataset
from retriever.envs.ravens.envs.environment import Environment

root = pathlib.Path.cwd()
ASSETS_PATH = root / "src" / "skills" / "cliport" / "cfg"
ASSETS_PATH = str(ASSETS_PATH)


@hydra.main(config_path=ASSETS_PATH, config_name="data")
def main(cfg):
    # Initialize environment and task.
    env = Environment(
        cfg["assets_root"],
        disp=cfg["disp"],
        shared_memory=cfg["shared_memory"],
        hz=480,
        record_cfg=cfg["record"],
    )
    task = tasks.names[cfg["task"]]()
    task.mode = cfg["mode"]
    record = cfg["record"]["save_video"]
    save_data = cfg["save_data"]

    # Initialize scripted oracle agents and dataset.
    agent = task.oracle(env)
    # agents = task.step_oracle(env)
    data_path = os.path.join(cfg["data_dir"], "{}-{}".format(cfg["task"], task.mode))
    dataset = RavensDataset(data_path, cfg, n_demos=0, augment=False)
    print(f"Saving to: {data_path}")
    print(f"Mode: {task.mode}")

    # Train seeds are even and val/test seeds are odd. Test seeds are offset by 10000
    seed = dataset.max_seed
    if seed < 0:
        if task.mode == "train":
            seed = -2
        elif task.mode == "val":  # NOTE: beware of increasing val set to >100
            seed = -1
        elif task.mode == "test":
            seed = -1 + 10000
        else:
            raise Exception("Invalid mode. Valid options: train, val, test")

    # Collect training data from oracle demonstrations.
    while dataset.n_episodes < cfg["n"]:
        episode, total_reward = [], 0
        seed += 2

        # Set seeds.
        np.random.seed(seed)
        random.seed(seed)

        print(
            "Oracle demo: {}/{} | Seed: {}".format(
                dataset.n_episodes + 1, cfg["n"], seed
            )
        )

        env.set_task(task)
        obs = env.reset()
        info = env.info
        # print(info)
        reward = 0
        # ########################################
        # top_down_obs, _, _ = env.render_camera(task.oracle_cams[0])
        # import imageio
        # imageio.imwrite('/home/huyingdong/cliport-master/images/demos_{}_seed{}.png'.format(cfg['task'], seed), top_down_obs)
        # ########################################

        # Unlikely, but a safety check to prevent leaks.
        if task.mode == "val" and seed > (-1 + 10000):
            raise Exception("!!! Seeds for val set will overlap with the test set !!!")

        # Start video recording (NOTE: super slow)
        if record:
            env.start_rec(f"{dataset.n_episodes+1:06d}")

        # high_level_lang_goal = info['high_level_lang_goal']
        # print(f'High Level Goal: {high_level_lang_goal}')

        # Rollout expert policy
        for _ in range(task.max_steps):
            act = agent.act(obs, info)
            # act = agents.act(obs, info['lang_goal'])
            episode.append((obs, act, reward, info))
            lang_goal = info["lang_goal"]
            obs, reward, done, info = env.step(act)
            # success = info['success']
            total_reward += reward
            print(
                f"Total Reward: {total_reward:.3f} | Done: {done} | Goal: {lang_goal}"
            )
            # print(f'Total Reward: {total_reward:.3f} | Done: {done} | Success: {success} | Goal: {lang_goal}')
            if done:
                break
        episode.append((obs, None, reward, info))

        # End video recording
        if record:
            env.end_rec()

        # Only save completed demonstrations.
        if save_data and total_reward > 0.99:
            dataset.add(seed, episode)


if __name__ == "__main__":
    main()
