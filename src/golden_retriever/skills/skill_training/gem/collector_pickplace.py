# Adopted from Mingxi, Haojie, Xupeng, ...

import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from retriever.robots.ur5_hhl.env import Env
from retriever.robots.ur5_hhl.utils import demo_util_pp as demo_util

# sys.path.append("./")
# sys.path.append("..")
# sys.path.append("/home/ur5/rgbd_grasp_ws/src/helping_hands_rl_ur5/LEPP")


# def setup_ros_dependencies():
#     global rospy, tf
#     import rospy
#     import tf


def moveAndPlace(env):
    place_pose_1 = [
        -0.37496883,
        -1.85229093,
        1.99809742,
        -1.71823579,
        -1.56919986,
        -0.37594825,
    ]
    place_pose_2 = [
        0.79398787,
        -1.03516657,
        1.43296528,
        -1.96834976,
        -1.57040912,
        0.79418302,
    ]
    env.ur5.moveToJ(place_pose_1)
    env.ur5.waitUntilNotMoving()
    env.ur5.moveToJ(place_pose_2)
    env.ur5.gripper.openGripper()


if __name__ == "__main__":
    # setup_ros_dependencies()
    import rospy
    import tf

    rospy.init_node("image_proxy")

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

    Z_MIN_ROBOT = -0.1621
    pre_pick_height = Z_MIN_ROBOT + 0.3
    pick_height = Z_MIN_ROBOT + 0.0
    pre_place_height = Z_MIN_ROBOT + 0.3
    place_height = Z_MIN_ROBOT + 0.2

    action_sequence = "pxyzr"
    obs_source = "raw"
    env = Env(
        ws_center=ws_center, action_sequence=action_sequence, obs_source=obs_source
    )

    tf_listener = tf.TransformListener()
    rospy.sleep(2)
    # env.ur5.moveToHome()
    # env.ur5.gripper.openGripper()
    env.ur5.gripper.openGripper()
    pixel_small_reso, pixel_big_reso = (
        env.cloud_proxy.img_width,
        env.cloud_proxy.img_height,
    )
    # Testing Observations and Image, to be deleted once done all the jobs
    instruction = "pick object and place into object"
    (
        rgbd1,
        rgbd2,
        rgbd3,
        intrinsic1,
        intrinsic2,
        intrinsic3,
        extrinsic1,
        extrinsic2,
        extrinsic3,
    ) = env.cloud_proxy.get_multi_obs()
    # fig, ax = plt.subplots(3,2)
    # ax[0][0].imshow(rgbd1[...,:3]/255)
    # ax[1][0].imshow(np.rot90(rgbd2[180:476,500:730,:3]/255, 2))
    # ax[2][0].imshow(rgbd3[...,:3]/255)
    # ax[0][1].imshow(rgbd1[...,3])
    # ax[1][1].imshow(np.rot90(rgbd2[180:476,500:730,3], 2))
    # ax[2][1].imshow(rgbd3[...,3])
    plt.imshow(np.rot90(rgbd2[180:476, 500:730, :3] / 255, 2))
    plt.show(block=False)
    plt.pause(1)
    # depth, rgb, clip_feature_pick, clip_feature_place = env.cloud_proxy.getClipObs(instruction, parsing=True)
    # print(f"RGB shape is :{rgb.shape}")
    # fig, ax = plt.subplots(3,2)
    # ax[0][0].imshow(depth)
    # ax[1][0].imshow((rgb/255+clip_feature_pick[..., None])/2)
    # ax[2][0].imshow((rgb/255+clip_feature_place[..., None])/2)
    # ax[1][1].imshow(clip_feature_pick)
    # ax[2][1].imshow(clip_feature_place)
    plt.show(block=False)

    # Testing ends
    current_time = time.strftime("%Y,%m,%d,%H,%M,%S").replace(",", "-")
    file_name = f"demo_{current_time}"
    demos = []
    info = {
        "intrinsic1": intrinsic1,
        "intrinsic2": intrinsic2,
        "intrinsic3": intrinsic3,
        "extrinsic1": extrinsic1,
        "extrinsic2": extrinsic2,
        "extrinsic3": extrinsic3,
    }
    demos.append(info)
    # demos = np.load('add the path of file that needs modification', allow_pickle=True)
    num_demos = 0
    collection_ongoing = True
    while collection_ongoing == True:
        episode_ongoing = True
        episode = []
        while episode_ongoing:
            env.ur5.moveToHome()
            instruction = input(
                "Type in pick&place instruction and Enter to take photo:"
            )
            # depth, rgb, clip_feature_pick, clip_feature_place = env.cloud_proxy.getClipObs(instruction, parsing=True)
            rgbd1, rgbd2, rgbd3, _, _, _, _, _, _ = env.cloud_proxy.get_multi_obs()
            # ax[0][0].imshow(rgbd1[...,:3]/255)
            # ax[1][0].imshow(np.rot90(rgbd2[180:476,500:730,:3]/255, 2))
            # ax[2][0].imshow(rgbd3[...,:3]/255)
            # ax[0][1].imshow(rgbd1[...,3])
            # ax[1][1].imshow(np.rot90(rgbd2[180:476,500:730,3], 2))
            # ax[2][1].imshow(rgbd3[...,3])
            plt.imshow(np.rot90(rgbd2[180:476, 500:730, :3] / 255, 2))
            plt.show(block=False)
            plt.pause(1)
            # ax[0][0].imshow(depth)
            # ax[1][0].imshow((rgb/255+clip_feature_pick[..., None])/2)
            # ax[2][0].imshow((rgb/255+clip_feature_place[..., None])/2)
            # ax[1][1].imshow(clip_feature_pick)
            # ax[2][1].imshow(clip_feature_place)
            # plt.show(block=False)
            # plt.pause(1)
            input("Move the arm to picking pos and press ENTER")
            xyz = env.ur5.tool_position.copy()
            p0_theta = env.ur5.joint_values.copy()[-1]
            quat0 = env.ur5.tool_quat.copy()
            # yellow
            # pentagon
            # block
            pose0 = np.concatenate((xyz, env.ur5.joint_values.copy()))

            xyz0_wrt_base, quat0_wrt_base = tf_listener.lookupTransform(
                "/base_link", "/tool0_controller", rospy.Time(0)
            )
            print(f"grasping pose is at {pose0}")
            print(
                f"grasping pose with respect to base is at {xyz0_wrt_base,quat0_wrt_base}"
            )

            labelled_x_pick, labelled_y_pick = demo_util.xy2pixel(
                [xyz[0], xyz[1]], pixel_small_reso, pixel_big_reso
            )
            # labelled_rgb = rgb.astype(int)
            # labelled_rgb[labelled_x_pick-2: labelled_x_pick+2, labelled_y_pick-2: labelled_y_pick+2] = 255
            # ax[1][0].imshow((labelled_rgb/255+clip_feature_pick[..., None])/2)
            # ax[2][0].imshow((labelled_rgb/255+clip_feature_place[..., None])/2)
            # ax[1][1].imshow(clip_feature_pick)
            # ax[2][1].imshow(clip_feature_place)
            # plt.show(block=False)
            # plt.pause(1)
            env.ur5.gripper.closeGripper()

            # This is the command under debug mode
            # x0, y0 = demo_util.pixel2xy([40, 75], pixel_small_reso, pixel_big_reso)
            # env.ur5.moveToP(x0, y0, -0.1621 + 0.15, 0, 0, 1.57)

            plt.pause(1)
            place_obj = input("PLACING Please manually move to the grasping pose.\n")
            xyz = env.ur5.tool_position.copy()
            p1_theta = env.ur5.joint_values.copy()[-1]
            quat1 = env.ur5.tool_quat.copy()
            pose1 = np.concatenate((xyz, env.ur5.joint_values.copy()))

            xyz1_wrt_base, quat1_wrt_base = tf_listener.lookupTransform(
                "/base_link", "/tool0_controller", rospy.Time(0)
            )
            print(f"grasping pose is at {pose1}")
            print(
                f"grasping pose with respect to base is at {xyz1_wrt_base,quat1_wrt_base}"
            )

            labelled_x_place, labelled_y_place = demo_util.xy2pixel(
                [xyz[0], xyz[1]], pixel_small_reso, pixel_big_reso
            )
            # labelled_rgb = rgb.astype(int)yellow pentagon block
            # labelled_rgb[labelled_x_place - 2: labelled_x_place + 2, labelled_y_place - 2: labelled_y_place + 2] = 255
            # ax[1][0].imshow((labelled_rgb/255+clip_feature_pick[..., None])/2)
            # ax[2][0].imshow((labelled_rgb/255+clip_feature_place[..., None])/2)
            # ax[1][1].imshow(clip_feature_pick)
            # ax[2][1].imshow(clip_feature_place)
            plt.show(block=False)
            plt.pause(1)
            env.ur5.gripper.openGripper()
            # demo = {'depth': depth,
            #         'rgb': rgb,
            #         'clip_feature_pick':clip_feature_pick,
            #         'clip_feature_place':clip_feature_place,
            #         'p0': [labelled_x_pick, labelled_y_pick],
            #         'p0_theta': p0_theta,
            #         'pose0': pose0,
            #         'quat0': quat0,
            #         'p1': [labelled_x_place, labelled_y_place],
            #         'p1_theta': p1_theta,
            #         'pose1': pose1,
            #         'quat1': quat1,
            #         'instruction': instruction}
            demo = {
                "rgbd1": rgbd1,
                "rgbd2": rgbd2,
                "rgbd3": rgbd3,
                "p0": [labelled_x_pick, labelled_y_pick],
                "p0_theta": p0_theta,
                "pose0": pose0,
                "quat0": quat0,
                "xyz0_wrt_base": xyz0_wrt_base,
                "quat0_wrt_base": quat0_wrt_base,
                "p1": [labelled_x_place, labelled_y_place],
                "p1_theta": p1_theta,
                "pose1": pose1,
                "quat1": quat1,
                "xyz1_wrt_base": xyz1_wrt_base,
                "quat1_wrt_base": quat1_wrt_base,
                "instruction": instruction,
            }
            x = input(
                "Enter r to record this demo and continue episode;\n"
                "Enter a to abandon this demo and continue episode;\n"
                "Enter re to record this demo and start a new episode;\n"
                "Enter ae to abandon this demo and start a new episode;\n"
                "Enter rs to record this demo, save and end;\n"
                "Enter as to abandon this demo, save and end."
            )
            if x == "r":
                episode.append(demo)
                num_demos += 1
                env.ur5.moveToHome()
            elif x == "a":
                env.ur5.moveToHome()
            elif x == "re":
                episode.append(demo)
                num_demos += 1
                env.ur5.moveToHome()
                episode_ongoing = False
                if len(episode) > 0:
                    demos.append(episode)
            elif x == "ae":
                env.ur5.moveToHome()
                episode_ongoing = False
                if len(episode) > 0:
                    demos.append(episode)
            elif x == "rs":
                episode.append(demo)
                demos.append(episode)
                num_demos += 1
                env.ur5.moveToHome()
                episode_ongoing = False
                collection_ongoing = False
            elif x == "as":
                if len(episode) > 0:
                    demos.append(episode)
                episode_ongoing = False
                collection_ongoing = False
            else:
                env.ur5.moveToHome()

    if not os.path.exists("demo_rss"):
        os.makedirs("demo_rss")
    today_date = datetime.now().strftime("%Y%m%d")

    data_path = os.path.join("demo_rss", today_date)
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    file_path = os.path.join(data_path, file_name)
    np.save(file_path, demos)
    print(f"demos (total of {num_demos}) are saved at {os.getcwd()}/{file_name}")
