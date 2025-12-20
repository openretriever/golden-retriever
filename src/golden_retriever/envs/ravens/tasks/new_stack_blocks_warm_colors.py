"""Put Blocks in Bowl Task."""

import collections
import random
import re

import numpy as np
import pybullet as p

from retriever.utils.cliport import utils

from .task import Task


class StackBlocksWarmColors(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = (
            "pick up the {pick} block and place it on the {place} block"
        )
        self.task_completed_desc = "done stacking only the blocks of warm colors."

    def reset(self, env):
        super().reset(env)
        n_bowls = np.random.randint(1, 3)
        n_cool_blocks = 3
        n_warm_blocks = 3

        color_names = self.get_colors()
        cool_colors = ["blue", "green", "cyan", "purple"]
        warm_colors = ["red", "orange", "yellow", "pink"]
        selected_cool_color_names = random.sample(cool_colors, n_cool_blocks)
        selected_warm_color_names = random.sample(warm_colors, n_warm_blocks)
        selected_cool_colors = [utils.COLORS[cn] for cn in selected_cool_color_names]
        selected_warm_colors = [utils.COLORS[cn] for cn in selected_warm_color_names]
        bowl_color_names = random.sample(color_names, n_bowls)
        bowl_colors = [utils.COLORS[cn] for cn in bowl_color_names]

        self.blockbowl_affordance = {}
        for key, _ in utils.CORNER_OR_SIDE.items():
            self.blockbowl_affordance[key] = 1.0

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = "bowl/bowl.urdf"
        bowl_poses = []
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, "fixed")
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_colors[i] + [1])
            bowl_poses.append(bowl_pose)
            self.blockbowl_affordance[bowl_color_names[i] + " bowl"] = 1.0

        # Add blocks.
        warm_blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = "stacking/block.urdf"
        self.color2block_id = {}
        # Add warm blocks.
        for i in range(n_warm_blocks):
            block_pose = self.get_random_pose(env, block_size)
            if i == 0:
                base_pose = block_pose
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=selected_warm_colors[i] + [1])
            warm_blocks.append((block_id, (0, None)))
            self.color2block_id[selected_warm_color_names[i]] = block_id
            self.blockbowl_affordance[selected_warm_color_names[i] + " block"] = 1.0
        self.warm_blocks = warm_blocks

        place_height = [0.03 + (i + 1) * 0.04 for i in range(n_warm_blocks - 1)]
        place_poses = [np.array([0, 0, h]) for h in place_height]
        targs = [(utils.apply(base_pose, i), base_pose[1]) for i in place_poses]
        # Goal: stack warm blocks.
        for i in range(n_warm_blocks - 1):
            self.goals.append(
                (
                    [warm_blocks[i + 1]],
                    np.ones((1, 1)),
                    [targs[i]],
                    False,
                    True,
                    "pose",
                    None,
                    1 / (n_warm_blocks - 1),
                )
            )
            self.lang_goals.append(
                self.lang_template.format(
                    pick=selected_warm_color_names[i + 1],
                    place=selected_warm_color_names[i],
                )
            )

        # Add cool blocks.
        cool_blocks = []
        for i in range(n_cool_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=selected_cool_colors[i] + [1])
            cool_blocks.append((block_id, (0, None)))
            self.color2block_id[selected_cool_color_names[i]] = block_id
            self.blockbowl_affordance[selected_cool_color_names[i] + " block"] = 1.0
        self.cool_blocks = cool_blocks

        # Only one mistake allowed.
        self.max_steps = len(warm_blocks)

        self.high_level_lang_goal = "stack only the blocks of warm colors"

    def get_colors(self):
        all_colors = utils.ALL_COLORS
        return all_colors

    def success(self):
        warm_height_threshold = 0.01 + 0.04 * (len(self.warm_blocks) - 1)
        warm_block_heights = [
            p.getBasePositionAndOrientation(bid)[0][2] for bid, _ in self.warm_blocks
        ]
        warm_is_stacked = any([h >= warm_height_threshold for h in warm_block_heights])

        cool_block_heights = [
            p.getBasePositionAndOrientation(bid)[0][2] for bid, _ in self.cool_blocks
        ]
        cool_on_ground = all([h <= 0.03 for h in cool_block_heights])

        warm_not_on_cool = True
        for cool_bid, _ in self.cool_blocks:
            cool_pose = p.getBasePositionAndOrientation(cool_bid)
            for warm_bid, _ in self.warm_blocks:
                warm_pose = p.getBasePositionAndOrientation(warm_bid)
                if self.is_match(cool_pose, warm_pose, symmetry=0):
                    warm_not_on_cool = False
                    break
            if not warm_not_on_cool:
                break

        return warm_is_stacked and cool_on_ground and warm_not_on_cool

    def step_oracle(self, env):
        """Oracle agents."""
        OracleAgent = collections.namedtuple("OracleAgent", ["act"])

        def act(obs, language_goal):
            """Calculate action."""

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            cmap, hmap, obj_mask = self.get_true_image(env)

            all_colors = self.get_colors()
            color_pattern = r"\b(" + "|".join(all_colors) + r")\b"
            color_names = re.findall(color_pattern, language_goal)
            assert len(color_names) == 2, "Oracle needs two colors to stack blocks"
            pick_color = color_names[0]
            place_color = color_names[1]
            pick_block_id = self.color2block_id[pick_color]
            place_block_id = self.color2block_id[place_color]

            # pick pose
            pick_mask = np.uint8(obj_mask == pick_block_id)
            if np.sum(pick_mask) == 0:
                raise ValueError("Pick block not found in segmentation mask")
            pick_pix = utils.sample_gaussian_distribution(pick_mask)
            pick_pos = utils.pix_to_xyz(pick_pix, hmap, self.bounds, self.pix_size)
            pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

            # place pose
            place_obj_pose = p.getBasePositionAndOrientation(place_block_id)
            targ_position = (
                place_obj_pose[0][0],
                place_obj_pose[0][1],
                place_obj_pose[0][2] + 0.04,
            )
            targ_pose = (targ_position, place_obj_pose[1])

            obj_pose = p.getBasePositionAndOrientation(
                pick_block_id
            )  # pylint: disable=undefined-loop-variable
            if not self.sixdof:
                obj_euler = utils.quatXYZW_to_eulerXYZ(obj_pose[1])
                obj_quat = utils.eulerXYZ_to_quatXYZW((0, 0, obj_euler[2]))
                obj_pose = (obj_pose[0], obj_quat)
            world_to_pick = utils.invert(pick_pose)
            obj_to_pick = utils.multiply(world_to_pick, obj_pose)
            pick_to_obj = utils.invert(obj_to_pick)
            place_pose = utils.multiply(targ_pose, pick_to_obj)

            place_pose = (np.asarray(place_pose[0]), np.asarray(place_pose[1]))

            return {"pose0": pick_pose, "pose1": place_pose}

        return OracleAgent(act)
