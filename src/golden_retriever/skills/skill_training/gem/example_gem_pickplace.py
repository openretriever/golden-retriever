# Adopted from code by Haojie, Mingxi

import os
import time

import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

import src.robots.ur5_hhl.utils.demo_util_pp as demo_util
from retriever.robots.ur5_hhl.env import Env
from retriever.skill_training.gem.cliport import agents
from retriever.skill_training.gem.cliport.dataset_real import RealDataset
from retriever.skill_training.gem.lepp.dataset_tool import dataTool
from retriever.skill_training.gem.lepp.parser import parse_instruction


# TODO separate ROS dependencies loading
def setup_ros_dependencies():
    global rospy
    import rospy


def load_hydra_config(config_path):
    return OmegaConf.load(config_path)


def process_obs(rgb, depth):
    img = np.concatenate(
        (rgb, depth[Ellipsis, None], depth[Ellipsis, None], depth[Ellipsis, None]),
        axis=2,
    )
    # img = utils.preprocess(img, dist='real')
    return img


def rotatePixelCoordinate(image_shape: tuple, pixel_xy: np.array, rotate_angle: float):
    """
    We define x, y to be row and column respectively
    rotate_angle is in rad
    """
    image_shape = np.array(image_shape)
    image_center = image_shape[:2] // 2
    rotation_mat = np.array(
        [
            [np.cos(rotate_angle), -np.sin(rotate_angle)],
            [np.sin(rotate_angle), np.cos(rotate_angle)],
        ]
    )
    pixel_xy = (pixel_xy - image_center).reshape(2, 1)
    length = np.sqrt(pixel_xy[0] ** 2 + pixel_xy[1] ** 2)
    result = rotation_mat.dot(pixel_xy).reshape(
        2,
    ) + np.array([image_center[1], image_center[0]])
    result_x = np.clip(result[0], 0, image_shape[1])
    result_y = np.clip(result[1], 0, image_shape[0])
    return np.array([result_x, result_y]).astype(int)


def rotateImage90(image: np.array):
    return np.rot90(image)


def visualize(
    rgb,
    depth,
    p0,
    p1,
    p0_theta,
    p1_theta,
    clip_features_pick=None,
    clip_features_place=None,
):
    fig, ax = plt.subplots(1, 4, figsize=(15, 5), dpi=300)
    ax[0].imshow(rgb / 255)
    ax[0].axis("off")
    ax[1].imshow(depth)
    ax[1].axis("off")
    if clip_features_pick is not None:
        ax[2].imshow(clip_features_pick)
        ax[2].axis("off")
    if clip_features_place is not None:
        ax[3].imshow(clip_features_place)
        ax[3].axis("off")
    # ax[2].imshow(clip_feature_pick[..., 0])
    # ax[3].imshow(clip_feature_place[..., 0])
    p0_theta = (p0_theta + 2 * np.pi) % (2 * np.pi)
    # p1_theta = p0_theta + p1_theta
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


if __name__ == "__main__":
    # TODO load ROS dependencies
    setup_ros_dependencies()

    rospy.init_node("eval_pp")

    exp_path = "/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP/exps"
    tcfg = load_hydra_config(
        "/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP/cliport/cfg/train.yaml"
    )
    agent_name = "LEPP"
    pick_kernel_name = "unetl"
    place_kernel_name = "eunet"
    model_name = "unetl-score-vit-postLinearMul"

    # agent_name = 'cliport'
    # pick_kernel_name = None
    # place_kernel_name = None
    # model_name = None

    logit_out_channel = 3
    # task_name = 'block-pyramid-stacking-processed'
    # task_name = "pick-letter-on-color-plates-processed"
    # task_name = "pick-block-in-bowl-processed"
    # task_name = "multi-processed-large"
    task_name = "pick-part-in-brown-box-processed"
    image_text_ratio = 0.5
    n_demos = 20
    model_file = "steps=50000.pt"
    allow_move = True
    kernel_size = 60
    stride = 20
    kernel_size_image = 40

    # tcfg['lepp']['model_name'] = 'unetl-score-vit-postLinearAdd'
    tcfg["lepp"]["model_name"] = model_name
    tcfg["lepp"]["pick_kernel_name"] = pick_kernel_name
    tcfg["lepp"]["place_kernel_name"] = place_kernel_name
    # agent_name = 'cliport'
    # tcfg['lepp']['pick_kernel_name'] = 'none'
    # tcfg['lepp']['model_name'] = 'none'
    # tcfg['lepp']['place_kernel_name'] = 'none'

    tcfg["train"]["agent"] = agent_name
    tcfg["train"]["task"] = task_name
    # tcfg['train']['task'] = 'pick-part-in-brown-box-processed'
    tcfg["dataset"]["type"] = "realtable"
    tcfg["dataset"]["use_image_goal"] = True
    tcfg["lepp"]["logit_out_channel"] = logit_out_channel
    tcfg["train"]["n_demos"] = n_demos
    tcfg["dataset"]["image_text_ratio"] = image_text_ratio

    training_npy = (
        "/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP/data/"
        + tcfg["train"]["task"]
        + ".npy"
    )
    # training_npy = '/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP/data/demo_2024-01-18-14-45-53-processed.npy'
    # data_path = '/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP/data/'
    train_ds = RealDataset(training_npy, tcfg, n_demos=n_demos)
    agent = agents.names[agent_name](None, tcfg, None, None)
    path = os.path.join(
        exp_path,
        task_name + f"-n-train{tcfg['train']['n_demos']}",
        f"{agent_name}-{tcfg['lepp']['model_name']}-{logit_out_channel}-{pick_kernel_name}-{place_kernel_name}-ParTrue-TopdownFalse-CropTrue-Ratio0.5",
        "checkpoints",
    )
    # path = os.path.join(exp_path, task_name + f"-n-train{tcfg['train']['n_demos']}", agent_name + '-unetl-score-vit-postLinearMul-3-unetl-eunet-ParTrue-TopdownFalse-CropTrue',"checkpoints")
    # path = os.path.join(exp_path, "pick-part-in-brown-box-processed" + f"-n-train{tcfg['train']['n_demos']}",
    #                     agent_name, "checkpoints")
    # path = os.path.join(exp_path, task_name, )
    agent.load(os.path.join(path, model_file))

    # clip_processor = CLIP_processor()
    querytool = dataTool(train_ds, tcfg["train"]["task"], "val")

    global model, alg, action_sequence, in_hand_mode, workspace, max_z, min_z
    # ws_center = [-0.5539, 0.0298, -0.1625]
    ws_center = [-0.445, -0.079, -0.1625]
    workspace = np.asarray(
        [
            [ws_center[0] - 0.15, ws_center[0] + 0.15],
            [ws_center[1] - 0.15, ws_center[1] + 0.15],
            [ws_center[2], 0.50],
        ]
    )
    max_z = 0.12
    min_z = 0.02

    action_sequence = "pxyzr"
    obs_source = "raw"
    block_clip_processor = True
    if agent_name == "LEPP":
        block_clip_processor = False
    env = Env(
        ws_center=ws_center,
        action_sequence=action_sequence,
        obs_source=obs_source,
        block_clip_processor=block_clip_processor,
        kernel_size=kernel_size,
        stride=stride,
    )
    pixel_small_reso, pixel_big_reso = (
        env.cloud_proxy.img_width,
        env.cloud_proxy.img_height,
    )
    rospy.sleep(2)
    env.ur5.moveToHome()
    rospy.sleep(2)
    log_data = []

    while True:
        instruction = input("Type instruction and then Press Enter take photo...\n")
        depth, rgb, clip_feature_pick, clip_feature_place = env.cloud_proxy.getObs(
            instruction, block_clip_processor=block_clip_processor, task_name=task_name
        )

        pick_inst, place_inst = parse_instruction(task_name, instruction)
        if agent_name == "LEPP":
            clip_feature_pick = clip_feature_pick[..., None]
            clip_feature_place = clip_feature_place[..., None]
            pick_text_feature = env.cloud_proxy.clip_processor.get_clip_text_feature(
                pick_inst
            )
            place_text_feature = env.cloud_proxy.clip_processor.get_clip_text_feature(
                place_inst
            )
            pick_crop, pick_similarity = querytool.query_crop(pick_text_feature)
            place_crop, place_similarity = querytool.query_crop(place_text_feature)
            # pick_crop = None
            # place_crop = None

            if pick_crop is not None:
                (
                    _,
                    _,
                    clip_feature_pick_crop,
                    _,
                ) = env.cloud_proxy.clip_processor.get_clip_feature_from_text_and_image(
                    rgb,
                    pick_inst,
                    pick_crop,
                    kernel_size=kernel_size_image,
                    stride=stride,
                )
                clip_feature_pick = (
                    clip_feature_pick * (1 - image_text_ratio)
                    + clip_feature_pick_crop * image_text_ratio
                )

            if place_crop is not None:
                (
                    _,
                    _,
                    clip_feature_place_crop,
                    _,
                ) = env.cloud_proxy.clip_processor.get_clip_feature_from_text_and_image(
                    rgb,
                    place_inst,
                    place_crop,
                    kernel_size=kernel_size_image,
                    stride=stride,
                )
                clip_feature_place = (
                    clip_feature_place * (1 - image_text_ratio)
                    + clip_feature_place_crop * image_text_ratio
                )

        img = process_obs(rgb, depth)
        # img = rotateImage90(img)
        # clip_feature_pick = rotateImage90(clip_feature_pick)
        # clip_feature_place = rotateImage90(clip_feature_place)
        obs = {
            "img": img,
            "clip_pick": clip_feature_pick,
            "clip_place": clip_feature_place,
        }
        info = {"lang_goal": instruction}

        act = agent.actReal(obs, info)

        p0_x, p0_y, p0_theta = act["pick"]
        p1_x, p1_y, p1_theta = act["place"]
        if agent_name == "LEPP":
            visualize(
                img[..., :3],
                img[..., 3],
                [p0_x, p0_y],
                [p1_x, p1_y],
                p0_theta,
                p1_theta,
                clip_feature_pick[..., 0],
                clip_feature_place[..., 0],
            )
            log_data.append(
                {
                    "rgbd": img,
                    "p0": act["pick"],
                    "p1": act["place"],
                    "clip_feature_pick": clip_feature_pick[..., 0],
                    "clip_feature_place": clip_feature_place[..., 0],
                }
            )
        else:
            visualize(
                img[..., :3],
                img[..., 3],
                [p0_x, p0_y],
                [p1_x, p1_y],
                p0_theta,
                p1_theta,
            )
            log_data.append(
                {
                    "rgbd": img,
                    "p0": act["pick"],
                    "p1": act["place"],
                }
            )

        Z_MIN_ROBOT = -0.1671
        pre_pick_height = Z_MIN_ROBOT + 0.3
        relative_pick_height = (
            depth[p0_x - 5 : p0_x + 5, p0_y - 5 : p0_y + 5].mean() - 0.09
        )
        pick_height = Z_MIN_ROBOT + np.clip(relative_pick_height, 0, 0.3)
        pre_place_height = Z_MIN_ROBOT + 0.30
        # place_height = Z_MIN_ROBOT + 0.2
        pixel_range = 10
        pixel_range_start = 5
        relative_place_height = np.sort(
            depth[
                p1_x - pixel_range : p1_x + pixel_range,
                p1_y - pixel_range : p1_y + pixel_range,
            ]
        )
        relative_place_height = relative_place_height[:5].mean()
        if " on " in instruction:
            if relative_place_height < 0.07:
                relative_place_height = relative_place_height - 0.002
            elif relative_place_height > 0.07 and relative_place_height < 0.1:
                relative_place_height = relative_place_height - 0.002
            else:
                relative_place_height = relative_place_height - 0.002
            place_height = Z_MIN_ROBOT + np.clip(relative_place_height, -0.001, 0.3)
        else:
            relative_place_height = relative_place_height + 0.08
            place_height = Z_MIN_ROBOT + 0.1

        # p0_x, p0_y = rotatePixelCoordinate(img.shape, np.array([p0_x, p0_y]), -np.pi/2)
        # p1_x, p1_y = rotatePixelCoordinate(img.shape, np.array([p1_x, p1_y]), -np.pi/2)
        # p0_theta = p0_theta - np.pi/2

        x0, y0 = demo_util.pixel2xy([p0_x, p0_y], pixel_small_reso, pixel_big_reso)
        x1, y1 = demo_util.pixel2xy([p1_x, p1_y], pixel_small_reso, pixel_big_reso)
        # p0_theta = p0_theta
        p0_theta = (p0_theta + 2 * np.pi) % (2 * np.pi)
        p1_theta = (p0_theta + p1_theta + 2 * np.pi) % (2 * np.pi)
        # p1_theta = p1_theta + 1.57

        if allow_move:
            env.ur5.moveToP(x0, y0, pre_pick_height, 0, 0, p0_theta)
            env.ur5.moveToP(x0, y0, pick_height, 0, 0, p0_theta)
            env.ur5.gripper.closeGripper()
            time.sleep(1)

            env.ur5.moveToP(x0, y0, pre_place_height, 0, 0, p0_theta)
            env.ur5.moveToP(x1, y1, pre_place_height, 0, 0, p1_theta)
            env.ur5.moveToP(x1, y1, place_height, 0, 0, p1_theta)
            env.ur5.gripper.openGripper()

            time.sleep(1)
            env.ur5.moveToP(x1, y1, 0.3, 0, 0, p1_theta)

            env.ur5.moveToHome()

        confirmation = input("q to quit; c to continue.")
        if confirmation == "q":
            break

        np.save("log_data_banana.npy", log_data)
