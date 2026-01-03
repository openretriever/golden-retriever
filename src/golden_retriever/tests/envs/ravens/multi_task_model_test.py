import os
import pathlib
import random

import numpy as np
from cliport import agents, tasks
from cliport.utils import utils

from retriever.envs.ravens.dataset import RavensDataset
from retriever.envs.ravens.envs.environment import Environment

train_demos = 1000  # number training demonstrations used to train agents
n_eval = 3  # number of evaluation instances
mode = "val"  # val or test
save_video = False

agent_name = "cliport"
model_task = "stack-block-pyramid-seq-seen-colors"  # multi-task agents conditioned with language goals
model_folder = "src/golden_retriever/scripts/cliport_quickstart"  # path to pre-trained checkpoint
ckpt_name = "last.ckpt"  # name of checkpoint to load

### Uncomment the task you want to evaluate on ###
eval_task = "stack-blocks"
# eval_task = 'put-blocks-on-corner-side'
# eval_task = 'put-blocks-matching-colors'
# eval_task = 'put-blocks-mismatched-colors'
# eval_task = 'put-blocks-different-corners'
# eval_task = 'stack-blocks-cool-colors'
# eval_task = 'stack-blocks-warm-colors'
# eval_task = 'sort-primary-color-blocks'

root_dir = str(pathlib.Path.cwd()) + "/"
assets_root = os.path.join(root_dir, "src/envs/ravens/envs/assets/")
config_file = "eval.yaml"

vcfg = utils.load_hydra_config(
    os.path.join(root_dir, f"src/skills/cliport/cfg/{config_file}")
)
vcfg["data_dir"] = os.path.join(root_dir, "data")
vcfg["mode"] = mode

vcfg["model_task"] = model_task
vcfg["eval_task"] = eval_task
vcfg["agent"] = agent_name

# Model and training config paths
model_path = os.path.join(root_dir, model_folder)
vcfg[
    "train_config"
] = f"{model_path}/{vcfg['model_task']}-{vcfg['agent']}-n{train_demos}-train/.hydra/config.yaml"
vcfg[
    "model_path"
] = f"{model_path}/{vcfg['model_task']}-{vcfg['agent']}-n{train_demos}-train/checkpoints/"

tcfg = utils.load_hydra_config(vcfg["train_config"])

# Load dataset
ds = RavensDataset(
    os.path.join(vcfg["data_dir"], f'{vcfg["eval_task"]}-{vcfg["mode"]}'),
    tcfg,
    n_demos=n_eval,
    augment=False,
)

eval_run = 0
name = "{}-{}-{}-{}".format(vcfg["eval_task"], vcfg["agent"], n_eval, eval_run)
print(f"\nEval ID: {name}\n")

# Initialize agents
utils.set_seed(eval_run, torch=True)
agent = agents.names[vcfg["agent"]](name, tcfg, None, ds)

# Load checkpoint
ckpt_path = os.path.join(vcfg["model_path"], ckpt_name)
print(f"\nLoading checkpoint: {ckpt_path}")
agent.load(ckpt_path)

record_cfg = {
    "save_video": save_video,
    "save_video_path": root_dir + "images",
    "add_text": True,
    "fps": 20,
    "video_height": 640,
    "video_width": 720,
}

# Initialize environment and task.
env = Environment(
    assets_root, disp=False, shared_memory=False, hz=480, record_cfg=record_cfg
)

episode = 0
num_eval_instances = min(n_eval, ds.n_episodes)

for i in range(num_eval_instances):
    print(f"\nEvaluation Instance: {i + 1}/{num_eval_instances}")

    # Load episode
    episode, seed = ds.load(i)
    goal = episode[-1]
    total_reward = 0
    np.random.seed(seed)
    random.seed(seed)

    # Set task
    task_name = vcfg["eval_task"]
    task = tasks.names[task_name]()
    task.mode = mode

    # Set environment
    env.seed(seed)
    env.set_task(task)
    obs = env.reset()
    info = env.info
    reward = 0

    step = 0
    done = False

    if save_video:
        env.start_rec(f"{model_task}-{task_name}-seed{seed}")

    # Rollout
    while (step <= task.max_steps) and not done:
        print(f"Step: {step + 1} ({task.max_steps} max)")

        # if step == 0:
        #     info['lang_goal'] = 'pick up the white block and place it on the bottom left corner'
        # elif step == 1:
        #     info['lang_goal'] = 'pick up the blue block and place it on the top right corner'
        # elif step == 2:
        #     info['lang_goal'] = 'pick up the orange block and place it on the top left corner'
        # elif step == 3:
        #     info['lang_goal'] = 'pick up the purple block and place it on the bottom right corner'

        # Get action predictions
        action = agent.act(obs, info, goal=None)
        lang_goal = info["lang_goal"]
        obs, reward, done, info = env.step(action)
        success = info["success"]
        total_reward += reward
        print(
            f"Total Reward: {total_reward:.3f} | Done: {done} | Success: {success} | Goal: {lang_goal}"
        )
        step += 1

    if done:
        print("Done. Success.")
    else:
        print("Max steps reached. Task failed.")

    if save_video:
        env.end_rec()
