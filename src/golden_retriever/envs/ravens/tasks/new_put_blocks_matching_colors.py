"""Put Blocks in Bowl Task."""

import collections
import random
import re

import numpy as np
import pybullet as p

from retriever.utils.cliport import utils

from .task import Task


class PutBlocksMatchingColors(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "pick up the {pick} block and place it on the {place} bowl"
        self.task_completed_desc = (
            "done putting all the blocks on the bowls with matching colors."
        )

    def reset(self, env):
        super().reset(env)
        n_bowls = np.random.randint(2, 4)  # 2, 3

        color_names = self.get_colors()
        selected_color_names = random.sample(color_names, n_bowls)
        colors = [utils.COLORS[cn] for cn in selected_color_names]

        self.blockbowl_affordance = {}
        for key, _ in utils.CORNER_OR_SIDE.items():
            self.blockbowl_affordance[key] = 1.0

        # Add bowls.
        bowl_poses = []
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = "bowl/bowl.urdf"
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = "stacking/block.urdf"
        self.color2block_id = {}
        self.color2bowl_id = {}

        for i in range(n_bowls):
            # Add bowl.
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, "fixed")
            p.changeVisualShape(bowl_id, -1, rgbaColor=colors[i] + [1])
            bowl_poses.append(bowl_pose)
            self.color2bowl_id[selected_color_names[i]] = bowl_id
            self.blockbowl_affordance[selected_color_names[i] + " bowl"] = 1.0

            # Add block.
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=colors[i] + [1])
            blocks.append((block_id, (0, None)))
            self.color2block_id[selected_color_names[i]] = block_id
            self.blockbowl_affordance[selected_color_names[i] + " block"] = 1.0

            # Add goal.
            self.goals.append(
                (
                    [blocks[i]],
                    np.ones((1, 1)),
                    [bowl_pose],
                    False,
                    True,
                    "pose",
                    None,
                    1 / n_bowls,
                )
            )
            self.lang_goals.append(
                self.lang_template.format(
                    pick=selected_color_names[i], place=selected_color_names[i]
                )
            )
        self.bowl_poses = bowl_poses
        self.blocks = blocks

        # Only one mistake allowed.
        self.max_steps = len(blocks) + 1

        # Add distractor bowls.
        distractor_color_names = [
            c for c in color_names if c not in selected_color_names
        ]
        random.shuffle(distractor_color_names)
        distractor_colors = [utils.COLORS[c] for c in distractor_color_names]

        n_distractors = 0
        max_distractos = np.random.randint(1, 2)
        while n_distractors < max_distractos:
            is_block = np.random.rand() > 0.5
            urdf = block_urdf if is_block else bowl_urdf
            size = block_size if is_block else bowl_size
            pose = self.get_random_pose(env, size)
            if not pose:
                continue
            obj_id = env.add_object(urdf, pose)
            color = distractor_colors[n_distractors]
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            if is_block:
                self.blockbowl_affordance[
                    distractor_color_names[n_distractors] + " block"
                ] = 1.0
            else:
                self.blockbowl_affordance[
                    distractor_color_names[n_distractors] + " bowl"
                ] = 1.0
            n_distractors += 1

        self.high_level_lang_goal = (
            "put all the blocks on the bowls with matching colors"
        )

    def get_colors(self):
        all_colors = utils.ALL_COLORS
        return all_colors

    def success(self):
        block_poses = [p.getBasePositionAndOrientation(bid) for bid, _ in self.blocks]
        matches = [
            self.is_match(block_poses[i], self.bowl_poses[i], symmetry=0)
            for i in range(len(block_poses))
        ]
        return all(matches)

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
            place_bowl_id = self.color2bowl_id[place_color]

            # pick pose
            pick_mask = np.uint8(obj_mask == pick_block_id)
            if np.sum(pick_mask) == 0:
                raise ValueError("Pick block not found in segmentation mask")
            pick_pix = utils.sample_gaussian_distribution(pick_mask)
            pick_pos = utils.pix_to_xyz(pick_pix, hmap, self.bounds, self.pix_size)
            pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

            # place pose
            targ_pose = p.getBasePositionAndOrientation(place_bowl_id)
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
