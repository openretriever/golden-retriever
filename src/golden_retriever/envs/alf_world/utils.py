import os
import sys

import alfworld.agents.environment as environment
import cv2
import gymnasium as gym
import numpy as np
import torch
import torchvision.transforms as T
import yaml

if sys.version_info >= (3, 8):
    from typing import Literal
else:
    from typing_extensions import Literal

from agents.aware_vlm_llm_agent import AwareVLMAgent
from agents.pure_vlm_agent import PureVLMAgent
from agents.unaware_vlm_llm_agent import UnawareVLMAgent

Model = Literal[
    "gpt-4", "gpt-3.5-turbo", "text-davinci-003", "meta-llama/Llama-3.2-3B-Instruct"
]

PREDEFINED_COLORS = [
    (255, 0, 0),
    (255, 47, 0),
    (255, 100, 0),
    (255, 147, 0),
    (255, 200, 0),
    (251, 245, 0),
    (208, 255, 0),
    (161, 255, 0),
    (108, 255, 0),
    (61, 255, 0),
    (7, 255, 0),
    (0, 255, 39),
    (0, 255, 92),
    (0, 255, 139),
    (0, 255, 192),
    (0, 255, 245),
    (0, 216, 255),
    (0, 163, 255),
    (0, 116, 255),
    (0, 63, 255),
    (0, 15, 255),
    (37, 0, 255),
    (84, 0, 255),
    (137, 0, 255),
    (184, 0, 255),
    (238, 0, 255),
    (255, 0, 224),
    (255, 0, 171),
    (255, 0, 124),
    (255, 0, 71),
]


def agent_factory(agent_type: str, llm_model, vlm_model, num_eval: int):
    if agent_type == "unaware_vlm_llm":
        return UnawareVLMAgent(llm_model, vlm_model, num_eval)
    elif agent_type == "aware_vlm_llm":
        return AwareVLMAgent(llm_model, vlm_model, num_eval)
    elif agent_type == "pure_vlm":
        return PureVLMAgent(vlm_model, num_eval)
    else:
        raise ValueError(f"Unsupported agent type: {agent_type}")


def load_config_file(path):
    assert os.path.exists(path), "Invalid config file"
    with open(path) as reader:
        config = yaml.safe_load(reader)
    return config


def draw_dashed_box(image, x, y, w, h, color):
    """
    Draw a dashed bounding box on the image.

    Args:
        image (numpy.ndarray): The image to draw on.
        x (int): X-coordinate of the top-left corner of the box.
        y (int): Y-coordinate of the top-left corner of the box.
        w (int): Width of the box.
        h (int): Height of the box.
        color (tuple): RGB color of the box.

    Returns:
        numpy.ndarray: The image with the dashed box drawn.
    """
    for j in range(x, x + w, 10):
        cv2.line(image, (j, y), (min(j + 5, x + w), y), color, 1)
        cv2.line(image, (j, y + h), (min(j + 5, x + w), y + h), color, 1)
    for k in range(y, y + h, 10):
        cv2.line(image, (x, k), (x, min(k + 5, y + h)), color, 1)
        cv2.line(image, (x + w, k), (x + w, min(k + 5, y + h)), color, 1)
    return image


def draw_label_with_background(image, text, position, box_color):
    """
    Draws a label with a black background on the image.

    Args:
        image (numpy.ndarray): The image to annotate.
        text (str): The text to draw on the image.
        position (tuple): The (x, y) position to draw the text.
        box_color (tuple): The color of the text.

    Returns:
        numpy.ndarray: The annotated image.
    """
    text_x, text_y = position
    # Calculate the size of the text
    label_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)[0]

    # Draw black background rectangle for the label
    cv2.rectangle(
        image,
        (text_x - 2, text_y - label_size[1] - 2),
        (text_x + label_size[0] + 2, text_y + 2),
        (0, 0, 0),
        -1,
    )

    # Draw label number on the image with the same color as the bounding box
    cv2.putText(
        image,
        text,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.3,
        box_color,
        1,
    )
    return image


def find_non_overlapping_position(
    initial_x,
    initial_y,
    label_width,
    label_height,
    existing_label_regions,
    image_width,
    image_height,
    max_shifts=10,
    shift_step=15,
):
    """
    Find a non-overlapping position for a label within image boundaries.

    Args:
        initial_x (int): Desired x-coordinate for the label.
        initial_y (int): Desired y-coordinate for the label.
        label_width (int): Width of the label.
        label_height (int): Height of the label.
        existing_label_regions (list of lists): List of existing label bounding boxes [x, y, w, h].
        image_width (int): Width of the image.
        image_height (int): Height of the image.
        max_shifts (int, optional): Maximum number of vertical shifts to attempt. Defaults to 10.
        shift_step (int, optional): Number of pixels to shift the label downward each time. Defaults to 15.

    Returns:
        tuple: A tuple (new_x, new_y) representing the non-overlapping position for the label.
    """

    # Initialize label position
    new_x = initial_x
    new_y = initial_y
    label_rect = [
        new_x,
        new_y - label_height - 2,
        label_width + 4,
        label_height + 4,
    ]  # Adding padding

    # Check if initial position is within image boundaries
    if (
        label_rect[0] < 0
        or label_rect[1] < 0
        or label_rect[0] + label_rect[2] > image_width
        or label_rect[1] + label_rect[3] > image_height
    ):
        # Adjust initial positions to be within boundaries
        new_x = min(max(new_x, 0), image_width - label_width - 4)
        new_y = min(max(new_y, label_height + 4), image_height - 1)
        label_rect = [
            new_x,
            new_y - label_height - 2,
            label_width + 4,
            label_height + 4,
        ]

    # Attempt to find a non-overlapping position
    for shift in range(max_shifts):
        if not is_overlap(label_rect, existing_label_regions):
            # Ensure the label remains within image boundaries after shifting
            if (
                label_rect[0] >= 0
                and label_rect[1] >= 0
                and label_rect[0] + label_rect[2] <= image_width
                and label_rect[1] + label_rect[3] <= image_height
            ):
                return new_x, new_y
        # Shift the label downward
        new_y += shift_step
        label_rect = [
            new_x,
            new_y - label_height - 2,
            label_width + 4,
            label_height + 4,
        ]

        # If shifting downward goes beyond the image, try shifting upward
        if new_y + label_height + 2 > image_height:
            new_y = initial_y - (shift + 1) * shift_step
            label_rect = [
                new_x,
                new_y - label_height - 2,
                label_width + 4,
                label_height + 4,
            ]
            if new_y - label_height - 2 < 0:
                # Cannot shift upward without going out of bounds
                break

    # If all shifts cause overlaps or go out of boundaries, return the initial position (may overlap or be out of bounds)
    return new_x, new_y


# Function to check overlap
def is_overlap(new_rect, existing_rects):
    x_new, y_new, w_new, h_new = new_rect
    for rect in existing_rects:
        x_e, y_e, w_e, h_e = rect
        if (
            x_new < x_e + w_e
            and x_new + w_new > x_e
            and y_new < y_e + h_e
            and y_new + h_new > y_e
        ):
            return True
    return False


def get_label_position(
    label_index,
    x,
    y,
    label_regions,
    annotated_image,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    font_scale=0.5,
    font_thickness=1,
    label_padding=2,
):
    """
    Calculate a non-overlapping and boundary-aware position for a label.

    Args:
        label_index (int): The index of the label (used for text).
        x (int): The x-coordinate of the bounding box.
        y (int): The y-coordinate of the bounding box.
        label_regions (list of lists): Existing label regions to prevent overlap [x, y, w, h].
        annotated_image (numpy.ndarray): The image on which labels are being drawn.
        font (int, optional): Font type for the label text. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
        font_scale (float, optional): Font scale for the label text. Defaults to 0.5.
        font_thickness (int, optional): Font thickness for the label text. Defaults to 1.
        label_padding (int, optional): Padding around the label text. Defaults to 2.

    Returns:
        tuple: A tuple (new_label_x, new_label_y) representing the position for the label.
    """
    # Define label text and calculate its size
    label_text = str(label_index)
    (label_width, label_height), _ = cv2.getTextSize(
        label_text, font, font_scale, font_thickness
    )

    # Initial label position (top-left corner with some padding)
    initial_label_x = x
    initial_label_y = y - 5

    # Get image dimensions
    image_height, image_width, _ = annotated_image.shape

    # Find a non-overlapping position for the label
    new_label_x, new_label_y = find_non_overlapping_position(
        initial_label_x,
        initial_label_y,
        label_width,
        label_height,
        label_regions,
        image_width,
        image_height,
        max_shifts=10,
        shift_step=15,  # Adjust shift_step as needed
    )

    # Ensure text is within image boundaries
    new_label_x = max(
        label_padding, min(new_label_x, image_width - label_width - label_padding)
    )
    new_label_y = max(
        label_height + label_padding, min(new_label_y, image_height - label_padding)
    )

    # Define the label bounding box with padding
    label_rect = [
        new_label_x,
        new_label_y - label_height - label_padding,
        label_width + 2 * label_padding,
        label_height + 2 * label_padding,
    ]

    # Add the label_rect to existing label regions
    label_regions.append(label_rect)

    return new_label_x, new_label_y


def convert_obs_to_tensors(frames):
    # [H, W, C] -> [C, H, W]
    transform = T.Compose([T.ToTensor()])
    image_tensors = [transform(i) for i in frames]
    # [C, H, W] -> [H, W, C]
    # [0, 1] -> [0, 255]
    # RGB -> BGR ?
    for i in range(len(image_tensors)):
        image_tensors[i] = image_tensors[i].permute(1, 2, 0)
        image_tensors[i] *= 255
        image_tensors[i] = image_tensors[i].int()
        image_tensors[i] = image_tensors[i][:, :, [2, 1, 0]]
    image_tensors = torch.stack(image_tensors, dim=0)
    return image_tensors


def get_obs_image(env):
    current_frames = env.get_frames()

    return convert_obs_to_tensors(current_frames)


def is_object_visible_and_valid(object_id, controller):
    """
    Check if the object is either a receptacle or an object and is visible.

    Parameters:
    - object_id: The ID of the object to check.
    - controller: The controller object that contains receptacles and objects.

    Returns:
    - bool: True if the object is valid and visible, False otherwise.
    """
    obj_metadata = controller.get_obj_id_from_metadata(object_id)
    visible = obj_metadata and obj_metadata["visible"]
    return (
        object_id in controller.receptacles or object_id in controller.objects
    ) and visible


class AlfEnv(gym.Env):
    def __init__(self, config_file, train_eval="train", obs_type="segmentation"):
        config = load_config_file(config_file)
        env_type = config["env"]["type"]
        # env = getattr(environment, env_type)(config, train_eval='train')
        # env = getattr(environment, env_type)(config, train_eval='eval_out_of_distribution')
        env = getattr(environment, env_type)(config, train_eval=train_eval)
        self.env = env.init_env(batch_size=1)
        # Add the previous admissible commands for step
        self.prev_admissible_commands = None
        self.num_envs = 1
        self._last_obs_text: str = ""
        self._num_fail: int = 0
        self._is_reset: bool = False
        self.obs_type = obs_type

    def step(self, action):
        # action, legal_action = process_action(self.env, action, self.prev_admissible_commands)
        if action[0] in self.prev_admissible_commands or action[0] == " ":
            legal_action = True

        else:
            legal_action = False
            self._num_fail += 1
        obs, scores, dones, infos = self.env.step(action)
        infos["observation_text"] = obs
        infos["frame_description"] = self.env.envs[
            0
        ].controller.print_frame_desc_oracle(action[0])
        self._last_obs_text = obs[0]
        infos["inventory"] = self.env.envs[0].controller.print_inventory()
        if self._num_fail > 4:
            self._is_reset = True

        reward = compute_reward(infos, legal_action)
        self.prev_admissible_commands = list(infos["admissible_commands"])[0]
        return self._get_obs(infos), reward, dones, infos

    def reset(
        self,
        seed=42,
    ):
        self.env.seed(seed)
        obs, infos = self.env.reset()
        infos["observation_text"] = obs
        self._last_obs_text = obs[0]
        self.prev_admissible_commands = list(infos["admissible_commands"])[0]
        self._is_reset = False
        self._num_fail = 0
        return self._get_obs(infos), infos

    def _get_obs(self, infos):
        image = get_obs_image(self.env)
        if self.obs_type == "segmentation":
            cur_obs = self.overlay_segs(image, infos)
        elif self.obs_type == "bounding_box":
            cur_obs = self.draw_bbox(image, infos)
        elif self.obs_type == "both":
            cur_obs_segs = self.overlay_segs(image, infos)
            cur_obs_bbox = self.draw_bbox(image, infos)
            cur_obs = {
                "segs": cur_obs_segs,
                "bbox": cur_obs_bbox,
            }
        return cur_obs

    def get_task_objects(self):
        pddl_params = self.env.envs[0].traj_data["pddl_params"]
        objects = [pddl_params["object_target"].lower()]
        if pddl_params["toggle_target"]:
            objects.append(pddl_params["toggle_target"].lower())
        return objects

    def draw_bbox(self, obs, infos):
        """
        Draw bounding boxes on the original image based on segmentation masks.

        Args:
            obs (torch.int32): Original observation image.
            infos : (dict) info returned by the environment
        Returns:
            dict: A dictionary containing:
                - 'image' (tensor): The original image with bounding boxes drawn around detected objects
                - 'object_bindings' (str): Information about object IDs
        """
        controller = self.env.envs[0].controller
        inst_color_count, inst_color_to_object_id = controller.get_instance_seg()
        object_dict = {}

        # Convert torch tensor to numpy array and ensure the image has correct data type for OpenCV
        annotated_image = obs[0].numpy().astype(np.uint8).copy()

        i = 1
        color_seg = cv2.cvtColor(infos["instance_segs"][0], cv2.COLOR_RGB2BGR)
        color_index = 0

        # List to keep track of existing label regions
        label_regions = []

        for color, _ in inst_color_count.most_common():
            if color in inst_color_to_object_id:
                object_id = inst_color_to_object_id[color]
                if is_object_visible_and_valid(object_id, controller):
                    color_mask = np.array(color[::-1])
                    mask = cv2.inRange(color_seg, color_mask, color_mask)
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )

                    if contours:
                        # Find the largest contour which will be considered as the segmented object
                        largest_contour = max(contours, key=cv2.contourArea)
                        x, y, w, h = cv2.boundingRect(largest_contour)

                        # Generate a unique random color for each bounding box
                        box_color = PREDEFINED_COLORS[
                            color_index % len(PREDEFINED_COLORS)
                        ]
                        color_index += 1

                        # Draw dashed bounding box on the annotated image
                        annotated_image = draw_dashed_box(
                            annotated_image, x, y, w, h, box_color
                        )

                        new_label_x, new_label_y = get_label_position(
                            label_index=i,
                            x=x,
                            y=y,
                            label_regions=label_regions,
                            annotated_image=annotated_image,
                            font=cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale=0.5,
                            font_thickness=1,
                            label_padding=2,
                        )

                        # Draw the label with background
                        annotated_image = draw_label_with_background(
                            annotated_image,
                            str(i),
                            (new_label_x, new_label_y),
                            box_color,
                        )

                        if object_id in controller.receptacles:
                            object_dict[i] = {
                                "name": controller.receptacles[object_id]["num_id"],
                                "type": "Receptacle",
                                "bounding_box": [x, y, w, h],
                            }
                        elif object_id in controller.objects:
                            object_dict[i] = {
                                "name": controller.objects[object_id]["num_id"],
                                "type": "Object",
                                "bounding_box": [x, y, w, h],
                            }
                        i += 1

        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
        obj_binding = "Object Bindings: " + "\n".join(
            f"{obj_id}--{object_dict[obj_id]['name']}" for obj_id in object_dict
        )
        annotated_image = annotated_image / annotated_image.max()

        visual_obs = convert_obs_to_tensors([annotated_image])
        cur_obs = {"image": visual_obs[0, :, :, :], "object_bindings": obj_binding}
        return cur_obs

    def overlay_segs(self, obs, infos, alpha=0.35):
        """
        Overlay segmentation masks on the original image.

        Args:
            obs (torch.int32): Original observation image.
            infos : (dict) info returned by the environment.
            alpha : (float) transparency of the segmentation masks.
        Returns:
            dict: A dictionary containing:
                - 'image' (tensor): The original image with segmentation masks overlayed
                - 'object_bindings' (str): Information about object IDs
        """

        controller = self.env.envs[0].controller
        inst_color_count, inst_color_to_object_id = controller.get_instance_seg()
        annotated_image = obs[0].numpy().astype(np.uint8).copy()

        object_dict = {}
        i = 1
        color_segmentation = infos["instance_segs"]
        color_seg = cv2.cvtColor(color_segmentation[0], cv2.COLOR_RGB2BGR)
        annotated_image = (1 - alpha) * annotated_image + alpha * color_seg

        # Initialize list to keep track of existing label regions to prevent overlap
        label_regions = []
        for color, _ in inst_color_count.most_common():
            if color in inst_color_to_object_id:
                object_id = inst_color_to_object_id[color]
                if is_object_visible_and_valid(object_id, controller):
                    color_mask = np.array(color[::-1])
                    mask = cv2.inRange(color_seg, color_mask, color_mask)
                    contours, _ = cv2.findContours(
                        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                    )
                    if contours:
                        # Find the largest contour which will be considered as the segmented object
                        largest_contour = max(contours, key=cv2.contourArea)
                        M = cv2.moments(largest_contour)
                        if M["m00"] != 0:
                            cX = int(M["m10"] / M["m00"])
                            cY = int(M["m01"] / M["m00"])
                        else:
                            cX, cY = 0, 0  # Default position if contour area is zero
                            # Get label position using the new function
                        new_label_x, new_label_y = get_label_position(
                            label_index=i,
                            x=cX,
                            y=cY,
                            label_regions=label_regions,
                            annotated_image=annotated_image,
                            font=cv2.FONT_HERSHEY_SIMPLEX,
                            font_scale=0.5,
                            font_thickness=1,
                            label_padding=2,
                        )

                        annotated_image = draw_label_with_background(
                            annotated_image,
                            str(i),
                            (new_label_x, new_label_y),
                            # tuple(int(c) for c in color),
                            (255, 255, 255),
                        )
                    ## assign each seg with name
                    if object_id in controller.receptacles:
                        object_dict[i] = {
                            "name": controller.receptacles[object_id]["num_id"],
                            "cX": cX,
                            "cY": cY,
                            "type": "Receptacle",
                        }
                    elif object_id in controller.objects:
                        object_dict[i] = {
                            "name": controller.objects[object_id]["num_id"],
                            "cX": cX,
                            "cY": cY,
                            "type": "Object",
                        }
                    i = i + 1
        # color_segmentation = cv2.cvtColor(color_segmentation, cv2.COLOR_BGR2RGB)
        annotated_image = cv2.cvtColor(
            annotated_image.astype(np.uint8), cv2.COLOR_BGR2RGB
        )
        obj_binding = "Object Bindings: " + "\n".join(
            f"{obj_id}--{object_dict[obj_id]['name']}" for obj_id in object_dict
        )
        annotated_image = annotated_image / annotated_image.max()

        visual_obs = convert_obs_to_tensors([annotated_image])
        cur_obs = {"image": visual_obs[0, :, :, :], "object_bindings": obj_binding}
        return cur_obs

    def encode_object_locations(self, task_object):
        """
        Encode the locations of task-related objects into a human-readable format.

        Args:
            task_object: Target object type to look for

        Returns:
            list: List of strings describing object locations
        """
        controller = self.env.envs[0].controller
        objects = controller.env.last_event.metadata["objects"]

        # Initialize empty dictionary to store receptacle information
        # Each entry will map receptacle name -> dict with:
        #   num_obj: Number of target objects in this receptacle
        #   openable: Whether receptacle can be opened/closed
        #   isOpen: Current open/closed state
        receps = {}
        for obj in objects:
            if (
                obj["objectType"].lower() == task_object
                and obj["parentReceptacles"] is not None
                and obj["parentReceptacles"][0] in controller.receptacles
            ):
                recep_name = controller.receptacles[obj["parentReceptacles"][0]][
                    "num_id"
                ]
                receps[recep_name] = receps.get(
                    recep_name, {"num_obj": 0, "openable": None, "isOpen": None}
                )
                receps[recep_name]["num_obj"] = receps[recep_name]["num_obj"] + 1

        for k, v in receps.items():
            for obj in objects:
                if (
                    obj["objectId"] in controller.receptacles
                    and controller.receptacles[obj["objectId"]]["num_id"] == k
                ):
                    v["openable"] = obj["openable"]
                    v["isOpen"] = obj["isOpen"]

        location_info = []
        for k, v in receps.items():
            beverb = "are" if v["num_obj"] > 1 else "is"
            plural_s = "s" if v["num_obj"] > 1 else ""
            preposition = "in" if v["openable"] else "on"
            location_info.append(
                f'There {beverb} {v["num_obj"]} {task_object}{plural_s} {preposition} {k}'
            )
        location_info = ", ".join(location_info)
        return location_info


def compute_reward(infos, legal_action):
    # A function to compute the shaped reward for the alfworld environment
    # infos: the info returned by the environment
    # legal_action: a boolean value to indicate if the action is legal
    ## Tentative rewards: r = success_reward * 10 + goal_conditioned_r - 1*illegal_action
    reward = 50 * float(infos["won"][0]) + float(
        infos["goal_condition_success_rate"][0]
    )
    if not legal_action:
        # adding a reward penalty to illegal actions
        reward -= 1
    reward = [reward]
    return torch.tensor(reward)
