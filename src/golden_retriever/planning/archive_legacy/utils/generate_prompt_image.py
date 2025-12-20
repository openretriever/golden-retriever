import os
import random

import imageio
import numpy as np

from retriever.envs.ravens import tasks
from retriever.envs.ravens.envs.environment import Environment

# stack-blocks
# put-blocks-on-corner-side
# put-blocks-matching-colors

# put-letters-alphabetical-order
# spell-word
# separate-vowels
for task_name in ["separate-vowels"]:
    n_eval = 1
    save_video = False
    root_dir = "/home/freax/Documents/vilacliport"
    assets_root = os.path.join(root_dir, "cliport/environments/assets/")

    data_dir = "testdata-letters"
    save_dir = os.path.join(
        "/home/freax/Documents/vilacliport/VLM_planner", data_dir, "prompt"
    )
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    record_cfg = {
        "save_video": save_video,
        "save_video_path": "/home/freax/Documents/vilacliport/images",
        "add_text": True,
        "fps": 20,
        "video_height": 640,
        "video_width": 720,
    }

    env = Environment(
        assets_root, disp=False, shared_memory=False, hz=480, record_cfg=record_cfg
    )

    task = tasks.names[task_name]()
    task.mode = "test"

    if task_name == "put-blocks-on-corner-side":
        seed = 10000
    else:
        seed = 9999

    for i in range(n_eval):
        # Set seeds.
        seed += 2
        np.random.seed(seed)
        random.seed(seed)
        env.seed(seed)

        env.set_task(task)
        obs = env.reset()
        front_obs = obs["color"][0]  # front camera, 480 x 640 x 3
        top_down_obs, _, _ = env.render_camera(task.oracle_cams[0])
        info = env.info

        high_level_lang_goal = info["high_level_lang_goal"]
        # capitalize the first letter
        high_level_lang_goal = (
            high_level_lang_goal[0].upper() + high_level_lang_goal[1:]
        )
        high_level_lang_goal = "Task: " + high_level_lang_goal
        print(high_level_lang_goal)

        # save front_obs to episode_dir
        imageio.imwrite(
            os.path.join(save_dir, "{}-front_obs.png".format(task_name)), front_obs
        )
        # save top_down_obs to episode_dir
        imageio.imwrite(
            os.path.join(save_dir, "{}-top_down_obs.png".format(task_name)),
            top_down_obs,
        )
