"""Put Blocks in Bowl Task."""

import collections
import random
import re

import numpy as np
import pybullet as p

from retriever.utils.cliport import utils

from .task import Task


class SortPrimaryColorBlocks(Task):
    def __init__(self):
        super().__init__()
        self.max_steps = 10
        self.pos_eps = 0.05
        self.lang_template = "pick up the {pick} block and place it on the left side"
        self.task_completed_desc = (
            "done stacking the primary color blocks on the left side."
        )

    def reset(self, env):
        super().reset(env)
        n_bowls = np.random.randint(1, 4)
        n_primary_blocks = np.random.randint(2, 4)

        color_names = self.get_colors()
        bowl_color_names = random.sample(color_names, n_bowls)
        bowl_colors = [utils.COLORS[cn] for cn in bowl_color_names]

        primary_color_names = ["red", "green", "blue"]
        selected_primary_index = np.sort(
            np.random.choice([0, 1, 2], n_primary_blocks, replace=False)
        )
        selected_primary_color_names = [
            primary_color_names[i] for i in selected_primary_index
        ]
        selected_primary_colors = [
            utils.COLORS[cn] for cn in selected_primary_color_names
        ]

        left_side_pos = utils.CORNER_OR_SIDE["left side"]
        left_side_pos = (left_side_pos[0], left_side_pos[1], 0)

        self.blockbowl_affordance = {}
        for key, _ in utils.CORNER_OR_SIDE.items():
            self.blockbowl_affordance[key] = 1.0

        # Add bowls.
        bowl_size = (0.12, 0.12, 0)
        bowl_urdf = "bowl/bowl.urdf"
        bowl_poses = []
        for i in range(n_bowls):
            bowl_pose = self.get_random_pose(env, bowl_size, [left_side_pos])
            bowl_id = env.add_object(bowl_urdf, bowl_pose, "fixed")
            p.changeVisualShape(bowl_id, -1, rgbaColor=bowl_colors[i] + [1])
            bowl_poses.append(bowl_pose)
            self.blockbowl_affordance[bowl_color_names[i] + " bowl"] = 1.0

        # Add blocks.
        primary_blocks = []
        block_size = (0.04, 0.04, 0.04)
        block_urdf = "stacking/block.urdf"
        self.color2block_id = {}
        for i in range(n_primary_blocks):
            block_pose = self.get_random_pose(env, block_size, [left_side_pos])
            block_id = env.add_object(block_urdf, block_pose)
            p.changeVisualShape(
                block_id, -1, rgbaColor=selected_primary_colors[i] + [1]
            )
            primary_blocks.append((block_id, (0, None)))
            self.color2block_id[selected_primary_color_names[i]] = block_id
            self.blockbowl_affordance[selected_primary_color_names[i] + " block"] = 1.0
        self.primary_blocks = primary_blocks

        left_side_pose = (left_side_pos, (0, 0, 0, 1))
        self.left_side_pose = left_side_pose
        place_height = [0.03 + i * 0.04 for i in range(n_primary_blocks)]
        place_poses = [np.array([0, 0, h]) for h in place_height]
        targs = [
            (utils.apply(left_side_pose, i), left_side_pose[1]) for i in place_poses
        ]

        # Goal: sort primary color blocks to the left side.
        for i in range(n_primary_blocks):
            self.goals.append(
                (
                    [primary_blocks[i]],
                    np.ones((1, 1)),
                    [targs[i]],
                    False,
                    True,
                    "pose",
                    None,
                    1 / n_primary_blocks,
                )
            )
            self.lang_goals.append(
                self.lang_template.format(pick=selected_primary_color_names[i])
            )

        # Only one mistake allowed.
        self.max_steps = n_primary_blocks

        # Add distractor blocks.
        distractor_color_names = [
            cn for cn in color_names if cn not in primary_color_names
        ]
        random.shuffle(distractor_color_names)
        distractor_colors = [utils.COLORS[c] for c in distractor_color_names]

        n_distractors = 0
        max_distractos = np.random.randint(2, 4)
        while n_distractors < max_distractos:
            pose = self.get_random_pose(env, block_size, [left_side_pos])
            if not pose:
                continue
            obj_id = env.add_object(block_urdf, pose)
            color = distractor_colors[n_distractors]
            if not obj_id:
                continue
            p.changeVisualShape(obj_id, -1, rgbaColor=color + [1])
            self.color2block_id[distractor_color_names[n_distractors]] = obj_id
            self.blockbowl_affordance[
                distractor_color_names[n_distractors] + " block"
            ] = 1.0
            n_distractors += 1

        self.high_level_lang_goal = (
            "Stack only the blocks of RGB colors on the left side"
        )

    def get_colors(self):
        all_colors = utils.ALL_COLORS
        return all_colors

    def success(self):
        heights = [0.02 + i * 0.04 for i in range(len(self.primary_blocks))]
        place_heights = [np.array([0, 0, h]) for h in heights]
        place_poses = [
            (utils.apply(self.left_side_pose, i), self.left_side_pose[1])
            for i in place_heights
        ]
        primary_block_poses = [
            p.getBasePositionAndOrientation(bid) for bid, _ in self.primary_blocks
        ]
        matches = [
            self.is_match(primary_block_poses[i], place_poses[i], symmetry=0)
            for i in range(len(self.primary_blocks))
        ]
        return all(matches)

    def step_oracle(self, env):
        """Oracle agents."""
        OracleAgent = collections.namedtuple("OracleAgent", ["act"])

        def act(obs, language_goal):
            """Calculate action."""

            # Oracle uses perfect RGB-D orthographic images and segmentation masks.
            cmap, hmap, obj_mask = self.get_true_image(env)

            pattern = r"pick up the (.+?) block and place it on the (.+?)$"
            match = re.search(pattern, language_goal)
            assert match is not None, "Oracle could not parse goal"
            pick_color = match.group(1)
            pick_block_id = self.color2block_id[pick_color]
            place_location = match.group(2)
            if place_location in [
                "top left corner",
                "top side",
                "top right corner",
                "left side",
                "right side",
                "bottom right corner",
                "bottom side",
                "bottom left corner",
            ]:
                place_on_block = False
                corner_or_side = place_location
            else:
                place_on_block = True
                all_colors = self.get_colors()
                color_pattern = r"\b(" + "|".join(all_colors) + r")\b"
                color_names = re.findall(color_pattern, place_location)
                assert (
                    len(color_names) == 1
                ), "Oracle needs exactly one color in place location"
                place_color = color_names[0]
                place_block_id = self.color2block_id[place_color]

            # pick pose
            pick_mask = np.uint8(obj_mask == pick_block_id)
            if np.sum(pick_mask) == 0:
                raise ValueError("Pick block not found in segmentation mask")
            pick_pix = utils.sample_gaussian_distribution(pick_mask)
            pick_pos = utils.pix_to_xyz(pick_pix, hmap, self.bounds, self.pix_size)
            pick_pose = (np.asarray(pick_pos), np.asarray((0, 0, 0, 1)))

            # place pose
            if not place_on_block:
                corner_or_side_pos = utils.CORNER_OR_SIDE[corner_or_side]
                corner_or_side_pos = (corner_or_side_pos[0], corner_or_side_pos[1], 0)
                corner_or_side_pix = utils.xyz_to_pix(
                    corner_or_side_pos, self.bounds, self.pix_size
                )
                height = hmap[corner_or_side_pix[0], corner_or_side_pix[1]]
                targ_height = height + 0.03
                targ_pos = (corner_or_side_pos[0], corner_or_side_pos[1], targ_height)
                targ_pose = (np.asarray(targ_pos), np.asarray((0, 0, 0, 1)))
            else:
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
