"""Image dataset."""

import os

import matplotlib.pyplot as plt
import numpy as np
from cliport import tasks
from cliport.tasks import cameras
from cliport.utils import utils
from torch.utils.data import Dataset

MULTI_TASKS = [
    "pick-part-in-brown-box-processed.npy",
    "pick-letter-on-color-plates-processed.npy",
    "pick-block-in-bowl-processed.npy",
    "block-pyramid-stacking-processed.npy",
]

# See transporter.py, regression.py, dummy.py, task.py, etc.
PIXEL_SIZE = 0.003125
CAMERA_CONFIG = cameras.RealSenseD415.CONFIG
BOUNDS = np.array([[0.25, 0.75], [-0.5, 0.5], [0, 0.28]])

# Names as strings, REVERSE-sorted so longer (more specific) names are first.
TASK_NAMES = (tasks.names).keys()
TASK_NAMES = sorted(TASK_NAMES)[::-1]

# def parse_instruction(task, lang_goal):
#     if 'done' in lang_goal or 'solve' in lang_goal:
#         return lang_goal, lang_goal
#     else:
#         if "put-block-in-box" in task:
#             # pick: pick {color} block and place into the brown box
#             pick_goal = " ".join(lang_goal.split(' ')[1:3])
#             place_goal = "brown box"


#         return pick_goal, place_goal
def demo_scale_selection(task, filename, dataset):
    if "large" in task:
        if filename in [
            "pick-block-in-bowl-processed.npy",
            "block-pyramid-stacking-processed.npy",
        ]:
            dataset = dataset[:15]
            output = []
            for demo in dataset:
                for step in demo:
                    output.append([step])
            return output
        elif filename in ["pick-part-in-brown-box-processed.npy"]:
            return dataset
        elif filename in ["pick-letter-on-color-plates-processed.npy"]:
            return dataset
    elif "mid" in task:
        if filename in [
            "pick-block-in-bowl-processed.npy",
            "block-pyramid-stacking-processed.npy",
        ]:
            dataset = dataset[:15]
            output = []
            for demo in dataset:
                for step in demo:
                    output.append([step])
            return output
        elif filename in ["pick-part-in-brown-box-processed.npy"]:
            output = dataset[:60]
            return output
        elif filename in ["pick-letter-on-color-plates-processed.npy"]:
            output = dataset[:54]
            return output
    elif "small" in task:
        if filename in [
            "pick-block-in-bowl-processed.npy",
            "block-pyramid-stacking-processed.npy",
        ]:
            dataset = dataset[:5]
            output = []
            for demo in dataset:
                for step in demo:
                    output.append([step])
            return output
        elif filename in ["pick-part-in-brown-box-processed.npy"]:
            output = dataset[:20]
            return output
        elif filename in ["pick-letter-on-color-plates-processed.npy"]:
            output = dataset[:18]
            return output


class RealDataset(Dataset):
    """A simple image dataset class."""

    def __init__(self, path, cfg, n_demos=0, augment=False):
        """A simple RGB-D image dataset."""
        self.cfg = cfg
        self.aug_theta_sigma = (
            self.cfg["dataset"]["augment"]["theta_sigma"]
            if "augment" in self.cfg["dataset"]
            else 60
        )

        if "multi" in path:
            self.task = cfg["train"]["task"]
            path = os.path.split(path)[0]
            self.cache = []
            for filename in MULTI_TASKS:
                datapath = os.path.join(path, filename)
                dataset = np.load(datapath, allow_pickle=True).tolist()
                dataset = demo_scale_selection(self.task, filename, dataset)
                self.cache += dataset
            self.n_episodes = len(self.cache)
            print(f"there are {self.n_episodes} episodes in training set")
            self.sample_set = np.arange(0, self.n_episodes)

        else:
            self.task = path.split("/")[-1].split(".")[0]
            self.cache = np.load(path, allow_pickle=True)

            self.n_episodes = len(self.cache)
            self.n_demos = n_demos
            if self.n_demos > 0:
                # self.sample_set = np.random.choice(range(self.n_episodes), self.n_demos, False)
                self.sample_set = np.arange(0, self.n_demos)

        self.augment = augment

        self.use_image_goal = self.cfg["dataset"]["use_image_goal"]

        # self.CLIP_processor = CLIP_processor()

    def process_sample(self, datum, augment=True):
        depth = datum[
            "depth"
        ]  # must assume the datum['depth'] is already been transferred to height
        rgb = datum["rgb"]
        instruction = datum["instruction"]

        # feature_image_pp, _ = self.CLIP_processor.get_clip_feature(rgb, instruction)
        # pick_inst, place_inst = parse_instruction(self.task, instruction)
        # feature_image_pick, _ = self.CLIP_processor.get_clip_feature(rgb, pick_inst)
        # feature_image_place, _ = self.CLIP_processor.get_clip_feature(rgb, place_inst)
        # feature_image = np.concatenate([feature_image_pp, feature_image_pick, feature_image_place], axis=2)
        if self.use_image_goal:
            feature_image = datum["clip_features_crop"]
        else:
            feature_image = datum["clip_features"]

        img = np.concatenate(
            (rgb, depth[Ellipsis, None], depth[Ellipsis, None], depth[Ellipsis, None]),
            axis=2,
        )
        p0 = datum["p0"]
        p1 = datum["p1"]
        p0_theta = datum["p0_theta"]
        p1_theta = datum["p1_theta"]
        p1_theta = p1_theta - p0_theta

        p0_theta = (2 * np.pi + p0_theta) % (2 * np.pi)
        p1_theta = (2 * np.pi + p1_theta) % (2 * np.pi)

        # p1 theta is the difference (no change during transform)

        if augment:
            img, _, (p0, p1), perturb_params = utils.perturb(
                img, [p0, p1], theta_sigma=self.aug_theta_sigma
            )
            feature_image = utils.apply_perturbation(feature_image, perturb_params)
            p0_theta = p0_theta + (-perturb_params[0])

        sample = {
            "img": img,
            "p0": p0,
            "p0_theta": p0_theta,
            "p1": p1,
            "p1_theta": p1_theta,
            "perturb_params": perturb_params,
            "clip_features": feature_image,
            "lang_goal": instruction,
        }

        return sample

    def process_goal(self, goal, perturb_params):
        depth = goal["depth"]
        rgb = goal["rgb"]
        instruction = goal["instruction"]
        # feature_image = self.CLIP_processor.get_clip_feature(rgb, instruction)
        if self.use_image_goal:
            feature_image = goal["clip_features_crop"]
        else:
            feature_image = goal["clip_features"]
        img = np.concatenate(
            (rgb, depth[Ellipsis, None], depth[Ellipsis, None], depth[Ellipsis, None]),
            axis=2,
        )

        sample = {
            "img": img,
            "p0": goal["p0"],
            "p0_theta": goal["p0_theta"],
            "p1": goal["p1"],
            "p1_theta": goal["p1_theta"],
            "perturb_params": perturb_params,
            "clip_features": feature_image,
            "lang_goal": instruction,
        }

        return sample

    def get_clip_feature_image(self, obs, clip_feature, cam_config=None):
        pass

    def visualizeSample(self, sample: dict):
        p0, p1 = sample["p0"], sample["p1"]
        p0_theta, p1_theta = sample["p0_theta"], sample["p1_theta"]
        print(sample["lang_goal"])
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(sample["img"][..., :3].astype(int))
        ax[1].imshow(sample["img"][..., 3])
        # ax[2].imshow(clip_feature_pick[..., 0])
        # ax[3].imshow(clip_feature_place[..., 0])
        p0_theta = (p0_theta + 2 * np.pi) % (2 * np.pi)
        p1_theta = p0_theta + p1_theta
        print("row, column, rotz:", p0[0], p0[1])
        ax[0].plot(p0[1], p0[0], marker="o", color="green")
        ax[0].plot(p1[1], p1[0], marker="x", color="red")
        arrow_length = 30
        ax[0].arrow(
            p0[1],
            p0[0],
            arrow_length * np.cos(p0_theta),
            -arrow_length * np.sin(p0_theta),
            width=0.005,
            color="green",
        )
        ax[0].arrow(
            p1[1],
            p1[0],
            arrow_length * np.cos(p1_theta),
            -arrow_length * np.sin(p1_theta),
            width=0.005,
            color="red",
        )
        fig.canvas.draw()
        plt.show(block=False)
        plt.pause(1)

    def get_sample(self):
        # Choose random episode.
        if len(self.sample_set) > 0:
            episode_id = np.random.choice(self.sample_set)
        else:
            episode_id = np.random.choice(range(self.n_episodes))
        episode = self.cache[episode_id]

        # Is the task sequential like stack-block-pyramid-seq?
        is_sequential_task = "-seq" in self.task

        # Return random observation action pair (and goal) from episode.
        if len(episode) == 1:
            i = 0
            g = 0
        else:
            i = np.random.choice(range(len(episode) - 1))
            g = i + 1 if is_sequential_task else -1
        sample, goal = episode[i], episode[g]

        # Process sample.
        sample = self.process_sample(sample, augment=self.augment)
        goal = self.process_goal(goal, perturb_params=sample["perturb_params"])

        return sample, goal

    def get_curr_task(self):
        return self.task
