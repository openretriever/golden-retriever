"""Put Blocks in Bowl Task."""

import collections
import random
import re

import numpy as np
import pybullet as p

from golden_retriever.utils.cliport import utils

from .task import Task


class PutBlocksMismatchedColors(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "pick up the {pick} block and place it on the {place} bowl"
        self.task_completed_desc = (
            "done putting all the blocks on the bowls with mismatched colors."
        )

    def reset(self, env):
        super().reset(env)
        n_blocks = np.random.randint(2, 4)  # 2, 3
        n_bowls = n_blocks + 1

        color_names = self.get_colors()
        block_color_names = random.sample(color_names, n_blocks)
        block_colors = [utils.COLORS[cn] for cn in block_color_names]

        avaliable_color_names = [
            cn for cn in color_names if cn not in block_color_names
        ]
        bowl_color_names = block_color_names + random.sample(
            avaliable_color_names, n_bowls - n_blocks
        )
        bowl_colors = [utils.COLORS[cn] for cn in bowl_color_names]

        self.blockbowl_affordance = {}
        for key, _ in utils.CORNER_OR_SIDE.items():
            self.blockbowl_affordance[key] = 1.0

        # Add blocks.
        blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = "stacking/block.urdf"
        self.color2block_id = {}
        for i in range(n_blocks):
            block_pose = self.get_random_pose(env, block_size)
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(block_id, -1, rgbaColor=block_colors[i] + [1])
            blocks.append((block_id, (0, None)))
            self.color2block_id[block_color_names[i]] = block_id
            self.blockbowl_affordance[block_color_names[i] + " block"] = 1.0

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = "bowl/bowl.urdf"
        bowl_poses = []
        self.color2bowl_id = {}
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size)
            bowl_id = env.add_object(bowl_urdf, bowl_pose, "fixed")
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_colors[i] + [1])
            bowl_poses.append(bowl_pose)
            self.color2bowl_id[bowl_color_names[i]] = bowl_id
            self.blockbowl_affordance[bowl_color_names[i] + " bowl"] = 1.0

        # Goal: put the blocks in the bowls with mismatched colors
        targ_matrix = np.ones((len(blocks), len(bowl_poses)))
        for i in range(len(blocks)):
            targ_matrix[i, i] = 0
        self.targ_matrix = targ_matrix

        block_bowl_pairs = []
        block_bowl_color_pairs = []
        matches = targ_matrix.copy()
        for _ in range(n_blocks):
            nn_dists = []
            nn_targets = []
            for i in range(n_blocks):
                object_id, (symmetry, _) = blocks[i]
                xyz, _ = p.getBasePositionAndOrientation(object_id)
                targets_i = np.argwhere(matches[i, :]).reshape(-1)
                if len(targets_i) > 0:
                    targets_xyz = np.float32([bowl_poses[j][0] for j in targets_i])
                    dists = np.linalg.norm(
                        targets_xyz - np.float32(xyz).reshape(1, 3), axis=1
                    )
                    nn = np.argmin(dists)
                    nn_dists.append(dists[nn])
                    nn_targets.append(targets_i[nn])
                else:
                    nn_dists.append(0)
                    nn_targets.append(-1)
            order = np.argsort(nn_dists)[::-1]
            order = [i for i in order if nn_dists[i] > 0]
            block_idx = order[0]
            block_bowl_pairs.append(
                (blocks[block_idx], bowl_poses[nn_targets[block_idx]])
            )
            block_bowl_color_pairs.append(
                (block_color_names[block_idx], bowl_color_names[nn_targets[block_idx]])
            )
            matches[block_idx, :] = 0
            matches[:, nn_targets[block_idx]] = 0

        for i in range(len(block_bowl_pairs)):
            self.goals.append(
                (
                    [block_bowl_pairs[i][0]],
                    np.ones((1, 1)),
                    [block_bowl_pairs[i][1]],
                    False,
                    True,
                    "pose",
                    None,
                    1 / n_blocks,
                )
            )
            self.lang_goals.append(
                self.lang_template.format(
                    pick=block_bowl_color_pairs[i][0],
                    place=block_bowl_color_pairs[i][1],
                )
            )

        self.blocks = blocks
        self.bowl_poses = bowl_poses

        # Only one mistake allowed.
        self.max_steps = n_blocks + 1

        self.high_level_lang_goal = "put the blocks on the bowls with mismatched colors"

    def get_colors(self):
        all_colors = utils.ALL_COLORS
        return all_colors

    def success(self):
        matches = self.targ_matrix.copy()
        for i in range(len(self.blocks)):
            object_id, (symmetry, _) = self.blocks[i]
            pose = p.getBasePositionAndOrientation(object_id)
            targets_i = np.argwhere(matches[i, :]).reshape(-1)
            for j in targets_i:
                target_pose = self.bowl_poses[j]
                if self.is_match(pose, target_pose, symmetry):
                    matches[i, :] = 0
                    matches[:, j] = 0
        is_all_zero = np.all(matches == 0)
        return is_all_zero

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
