import os

import cv2

# from lepp.parser import parse_instruction
import lepp.transformations as transformation
import matplotlib.pyplot as plt
import numpy as np
import open3d
import scipy
import skimage.transform
import torch
import torch.nn.functional as F
from lepp.clip_preprocess import CLIP_processor


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
    clip_features_text,
    clip_features_image,
    clip_features,
):
    fig, ax = plt.subplots(2, 5)
    ax[0][0].imshow(rgb.astype(int))
    ax[0][0].set_title("RGB")
    ax[0][1].imshow(depth)
    ax[0][1].set_title("depth")
    ax[0][2].imshow(
        clip_features[
            ...,
            1:2,
        ]
    )
    ax[0][2].set_title("combined clip_features")
    ax[0][3].imshow(clip_features_text)
    ax[0][3].set_title("text clip_features")
    ax[0][4].imshow(clip_features_image)
    ax[0][4].set_title("image clip_features")

    ax[1][2].imshow(
        (
            clip_features[
                ...,
                1:2,
            ]
            * rgb
        ).astype(int)
    )
    ax[1][2].set_title("combined clip_features")
    ax[1][3].imshow((clip_features_text * rgb).astype(int))
    ax[1][3].set_title("text clip_features")
    ax[1][4].imshow((clip_features_image * rgb).astype(int))
    ax[1][4].set_title("image clip_features")
    # ax[2].imshow(clip_feature_pick[..., 0])
    # ax[3].imshow(clip_feature_place[..., 0])
    p0_theta = (p0_theta + 2 * np.pi) % (2 * np.pi)
    p1_theta = p0_theta + p1_theta
    # print('row, column, rotz:', p0[0], p0[1])
    ax[0][0].plot(p0[1], p0[0], marker="o", color="green")
    ax[0][0].plot(p1[1], p1[0], marker="x", color="red")
    arrow_length = 30
    ax[0][0].arrow(
        p0[1],
        p0[0],
        arrow_length * np.cos(p0_theta),
        -arrow_length * np.sin(p0_theta),
        width=0.005,
        color="green",
    )
    ax[0][0].arrow(
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


def get_crop(rgb, pixel_xy, clip_kernel_size):
    pad_size = clip_kernel_size // 2
    pad_rgb = (
        F.pad(
            input=torch.from_numpy(rgb).permute(2, 0, 1),
            pad=(pad_size, pad_size, pad_size, pad_size),
            mode="constant",
        )
        .permute(1, 2, 0)
        .numpy()
    )

    x, y = np.array(pixel_xy) + pad_size
    return pad_rgb[x - pad_size : x + pad_size, y - pad_size : y + pad_size, :]


def parse_instruction(instruction, task_name):
    if "pyramid" in task_name:
        # pick: pick {something} and place into {something}
        pick, place = instruction.split(" on ")
        pick_goal = " ".join(pick.split(" ")[1:-2])
        place_goal = " ".join(place.split(" ")[:])
    else:
        pick, place = instruction.split(" and ")
        pick_goal = " ".join(pick.split(" ")[1:])
        place_goal = " ".join(place.split(" ")[2:])
    return pick_goal, place_goal


class real_data_processor:
    def __init__(self, file_path, file_list, output_name) -> None:
        self.workspace = np.array([[0.22, 0.65], [-0.2, 0.37], [-0.3, 0.2]])
        self.center = self.workspace.mean(-1)
        self.x_size = self.workspace[0].max() - self.workspace[0].min()
        self.x_half = self.x_size / 2
        self.y_size = self.workspace[1].max() - self.workspace[1].min()
        self.y_half = self.y_size / 2
        self.z_min = self.workspace[2].min()

        self.clip_processor = CLIP_processor()
        self.kernel_size = 80
        self.stride = 20
        self.image_text_ratio = 0.5

        # self.query_tool = dataTool( self.train_ds, self.task, mode, MULTI_TASKS_list)

        self.img_width = 240
        self.img_height = 320
        self.z_table = -0.173
        self.rgb_value = [189, 152, 130]

        # self.raw_data_path = "/".join(raw_data_path.split('/')[:-1])
        # self.raw_file_name = self.raw_data_path.split('/')[-1].split('.')[0]
        self.file_path = file_path
        self.output_name = output_name

        self.demos = []
        for data_file in file_list:
            loading_path = os.path.join(file_path, data_file)
            data = np.load(loading_path, allow_pickle=True).tolist()
            self.demos = self.demos + data[1:]
        info = data[0]
        self.intrinsic1 = np.array(
            [
                [641.45025635, 0.0, 642.91943359],
                [0.0, 640.4543457, 358.3414917],
                [0.0, 0.0, 1.0],
            ]
        )
        self.intrinsic2 = np.array(
            [
                [607.78125, 0.0, 640.72650146],
                [0.0, 607.60565186, 366.64157104],
                [0.0, 0.0, 1.0],
            ]
        )
        self.intrinsic3 = np.array(
            [
                [640.62329102, 0.0, 642.87658691],
                [0.0, 639.61877441, 355.54577637],
                [0.0, 0.0, 1.0],
            ]
        )
        self.intrinsic3_sm = np.array(
            [
                [384.37402344, 0.0, 321.72595215],
                [0.0, 383.77124023, 237.32745361],
                [0.0, 0.0, 1.0],
            ]
        )
        self.extrinsic1 = np.array(
            [
                [-0.99847678, 0.02397427, -0.04969265, 0.43286359],
                [0.05229489, 0.69834904, -0.71384443, 0.65537134],
                [0.01758892, -0.71535575, -0.69853903, 0.32103107],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.extrinsic2 = np.array(
            [
                [0.99964383, -0.02175758, 0.01545371, 0.45690583],
                [-0.02122889, -0.99921026, -0.03358863, 0.05730241],
                [0.01617232, 0.0332486, -0.99931626, 0.91449623],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )
        self.extrinsic3 = np.array(
            [
                [0.99986521, 0.00128428, -0.01636822, 0.34889658],
                [0.01361058, -0.62240352, 0.78257818, -0.40142761],
                [-0.00918259, -0.78269547, -0.62233711, 0.2795286],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        self.table_to_topdown_height = self.extrinsic2[2, -1] + np.abs(self.z_table)
        # self.demos = data[1:]

    def correct_instruction(self, instruction):
        if "hammer" in instruction:
            instruction = instruction.replace("hammer", "plush toy hammer")
        if "bowl black part" in instruction:
            instruction = instruction.replace("bowl black part", "bowl dark blue part")
        return instruction

    def append_process_and_save(self, file_being_append):
        previous_demos = np.load(file_being_append, allow_pickle=True).tolist()
        demos = self.process_all_and_save()
        np.save(os.path.join(self.file_path, self.output_name), previous_demos + demos)

    def process_all_and_save(self):
        demos = []
        fig, ax = plt.subplots(3, 2)
        depths = []
        reds = []
        greens = []
        blues = []
        for i, current_demo in enumerate(self.demos):
            episode = []
            for j, current_episode in enumerate(current_demo):
                self.current_episode = current_episode
                depth = np.rot90(self.getDepthImage(2)[190:480, 515:725], 2)
                rgb = np.rot90(self.getRGBImage(2)[190:480, 515:725], 2)
                depth = skimage.transform.resize(
                    depth, (self.img_height, self.img_width)
                )
                rgb = skimage.transform.resize(
                    rgb, (self.img_height, self.img_width, 3)
                )
                depth = np.clip(self.table_to_topdown_height - depth, 0, 0.3)

                instruction = self.current_episode["instruction"]
                instruction = self.correct_instruction(instruction)
                pick_ins, place_ins = parse_instruction(instruction, self.output_name)
                _, _, clip_feature_pick, clip_feature_place = self.getClipObs(
                    instruction, parsing=True
                )
                pick_text_emb = self.clip_processor.get_clip_text_feature(pick_ins)
                place_text_emb = self.clip_processor.get_clip_text_feature(place_ins)
                print(instruction)

                p0_theta, p1_theta = (
                    transformation.euler_from_quaternion(
                        current_episode["quat0"], axes="szyx"
                    )[0],
                    transformation.euler_from_quaternion(
                        current_episode["quat1"], axes="szyx"
                    )[0],
                )
                ax[0][0].imshow(depth)
                ax[1][0].imshow((rgb / 255 + clip_feature_pick[..., None]) / 2)
                ax[1][0].plot(
                    current_episode["p0"][1],
                    current_episode["p0"][0],
                    marker="o",
                    color="green",
                )
                ax[1][1].imshow(clip_feature_pick)
                ax[2][0].imshow((rgb / 255 + clip_feature_place[..., None]) / 2)
                ax[1][0].plot(
                    current_episode["p1"][1],
                    current_episode["p1"][0],
                    marker="o",
                    color="red",
                )
                ax[2][1].imshow(clip_feature_place)
                arrow_length = 30
                ax[1][0].arrow(
                    current_episode["p0"][1],
                    current_episode["p0"][0],
                    arrow_length * np.cos(p0_theta),
                    -arrow_length * np.sin(p0_theta),
                    width=0.005,
                    color="green",
                )
                ax[1][0].arrow(
                    current_episode["p1"][1],
                    current_episode["p1"][0],
                    arrow_length * np.cos(p1_theta),
                    -arrow_length * np.sin(p1_theta),
                    width=0.005,
                    color="red",
                )
                plt.show(block=False)

                # clip_pp = (clip_feature_pick+clip_feature_place)/2
                clip_pp = np.max(
                    np.stack([clip_feature_pick, clip_feature_place]), axis=0
                )

                crop_kernel_size = 40
                # pad_rgb = F.pad(input=torch.from_numpy(episode[0]['rgb']).permute(2,0,1), pad=(pad_size, pad_size, pad_size, pad_size), mode='replicate').permute(1,2,0).numpy()
                pick_crop = get_crop(rgb, current_episode["p0"], crop_kernel_size)
                # pick_crop = episode[0]['rgb'][episode[0]['p0'][0]-pad_size:episode[0]['p0'][0]+pad_size, episode[0]['p0'][1]-pad_size:episode[0]['p0'][1]+pad_size, :]
                # place_crop = episode[0]['rgb'][episode[0]['p1'][0]-pad_size:episode[0]['p1'][0]+pad_size, episode[0]['p1'][1]-pad_size:episode[0]['p1'][1]+pad_size, :]
                place_crop = get_crop(rgb, current_episode["p1"], crop_kernel_size)
                clip_features = np.concatenate(
                    [
                        clip_pp[..., None],
                        clip_feature_pick[..., None],
                        clip_feature_place[..., None],
                    ],
                    axis=2,
                )

                # _, _, clip_feature_pick, clip_feature_place = self.getClipObs(instruction, parsing=True)
                clip_pp = np.max(
                    np.stack([clip_feature_pick, clip_feature_place]), axis=0
                )
                pick_feature_crop = (
                    self.clip_processor.get_clip_feature_from_text_and_image(
                        rgb, pick_ins, pick_crop
                    )[2]
                )
                place_feature_crop = (
                    self.clip_processor.get_clip_feature_from_text_and_image(
                        rgb, place_ins, place_crop
                    )[2]
                )
                clip_feature_pick = (
                    clip_feature_pick * (1 - self.image_text_ratio)
                    + pick_feature_crop[..., 0] * self.image_text_ratio
                )
                clip_feature_place = (
                    clip_feature_place * (1 - self.image_text_ratio)
                    + place_feature_crop[..., 0] * self.image_text_ratio
                )
                clip_features_crop = np.concatenate(
                    [
                        clip_pp[..., None],
                        clip_feature_pick[..., None],
                        clip_feature_place[..., None],
                    ],
                    axis=2,
                )
                ax[0][0].imshow(depth)
                ax[1][0].imshow((rgb / 255 + clip_feature_pick[..., None]) / 2)
                ax[1][0].plot(
                    current_episode["p0"][1],
                    current_episode["p0"][0],
                    marker="o",
                    color="green",
                )
                ax[1][1].imshow(clip_feature_pick)
                ax[2][0].imshow((rgb / 255 + clip_feature_place[..., None]) / 2)
                ax[1][0].plot(
                    current_episode["p1"][1],
                    current_episode["p1"][0],
                    marker="o",
                    color="red",
                )
                ax[2][1].imshow(clip_feature_place)
                arrow_length = 30
                ax[1][0].arrow(
                    current_episode["p0"][1],
                    current_episode["p0"][0],
                    arrow_length * np.cos(p0_theta),
                    -arrow_length * np.sin(p0_theta),
                    width=0.005,
                    color="green",
                )
                ax[1][0].arrow(
                    current_episode["p1"][1],
                    current_episode["p1"][0],
                    arrow_length * np.cos(p1_theta),
                    -arrow_length * np.sin(p1_theta),
                    width=0.005,
                    color="red",
                )
                plt.show(block=False)

                depths.append(depth)
                reds.append(rgb[..., 0])
                greens.append(rgb[..., 1])
                blues.append(rgb[..., 2])

                demo_processed = {
                    "depth": depth,
                    "rgb": rgb,
                    "clip_features": clip_features,
                    "clip_features_crop": clip_features_crop,
                    "pick_crop": pick_crop,
                    "place_crop": place_crop,
                    "pick_text_emb": pick_text_emb,
                    "place_text_emb": place_text_emb,
                    "p0": self.current_episode["p0"],
                    "p0_theta": p0_theta,
                    "pose0": self.current_episode["pose0"],
                    "quat0": self.current_episode["quat0"],
                    "p1": self.current_episode["p1"],
                    "p1_theta": p1_theta,
                    "pose1": self.current_episode["pose1"],
                    "quat1": self.current_episode["quat1"],
                    "instruction": instruction,
                }
                episode.append(demo_processed)
            demos.append(episode)

        depths = np.stack(depths)
        reds = np.stack(reds)
        greens = np.stack(greens)
        blues = np.stack(blues)
        print(f"Depth: mean{depths.mean()}, std{depths.std()}")
        print(f"reds: mean{reds.mean()}, std{reds.std()}")
        print(f"greens: mean{greens.mean()}, std{greens.std()}")
        print(f"blues: mean{blues.mean()}, std{blues.std()}")

        np.save(os.path.join(self.file_path, self.output_name), demos)

        return demos

    def getDepthImage(self, cam_id):
        if cam_id == 1:
            image = self.current_episode["rgbd1"][..., 3]
        if cam_id == 2:
            image = self.current_episode["rgbd2"][..., 3]
        if cam_id == 3:
            image = self.current_episode["rgbd3"][..., 3]

        return image

    def getRGBImage(self, cam_id):
        if cam_id == 1:
            image = self.current_episode["rgbd1"][..., :3]
        if cam_id == 2:
            image = self.current_episode["rgbd2"][..., :3]
        if cam_id == 3:
            image = self.current_episode["rgbd3"][..., :3]

        return image

    def get_pointcloud_from_depth(
        self, cam_id, instruction=None, visualize=False, image_cond=None
    ):
        depth = self.getDepthImage(cam_id)
        rgb = self.getRGBImage(cam_id)
        kernel_size = self.kernel_size
        stride = self.stride
        if cam_id == 1:
            intrinsics = self.intrinsic1
        if cam_id == 2:
            intrinsics = self.intrinsic2
        if cam_id == 3:
            if depth.shape[0] == 480:
                intrinsics = self.intrinsic3_sm
                kernel_size = kernel_size // 2
            else:
                intrinsics = self.intrinsic3

        if instruction is not None:
            if image_cond is not None:
                img_goal = image_cond
                (
                    clip_feature,
                    _,
                ) = self.clip_processor.get_clip_feature_from_text_and_image(
                    rgb, instruction, img_goal, kernel_size=kernel_size, stride=stride
                )
            else:
                clip_feature, _ = self.clip_processor.get_clip_feature(
                    rgb, instruction, kernel_size=kernel_size, stride=stride
                )

        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])
        if instruction is not None:
            points = np.float32(
                [
                    px,
                    py,
                    depth,
                    rgb[..., 0],
                    rgb[..., 1],
                    rgb[..., 2],
                    clip_feature[..., 0],
                ]
            ).transpose(1, 2, 0)
            cloud = points.reshape(-1, 7)
        else:
            points = np.float32(
                [px, py, depth, rgb[..., 0], rgb[..., 1], rgb[..., 2]]
            ).transpose(1, 2, 0)
            cloud = points.reshape(-1, 6)
        # z_constrain= (cloud[:,2]>0.1) & (cloud[:,2]<1.1)
        # cloud = cloud[z_constrain]
        if visualize:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(cloud[:, :3])
            pcd.colors = open3d.utility.Vector3dVector(cloud[:, 3:6] / 255)
            open3d.visualization.draw_geometries([pcd])
        # the output cloud is in rgb frame
        return cloud

    def transform_clip_cloud_to_base(self, cloud, cam_id):
        if cam_id == 1:
            proj_pose = self.extrinsic1
        if cam_id == 2:
            proj_pose = self.extrinsic2
        if cam_id == 3:
            proj_pose = self.extrinsic3
        cloud_RT_base = self.transform(cloud[:, :3], proj_pose)
        return np.concatenate([cloud_RT_base, cloud[:, 3:]], axis=1)

    def cloud_preprocess(self, cloud):
        cloud, rgb_clip = self.getFilteredPointCloud(cloud[:, :3], cloud[:, 3:])
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cloud)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
        cloud = np.asarray(cl.points)
        rgb_clip = rgb_clip[ind]
        cloud = np.concatenate([cloud, rgb_clip], axis=1)
        return cloud

    def clear_cache(self):
        self.depth1 = None
        self.depth2 = None
        self.depth3 = None
        self.rgbimg1 = None
        self.rgbimg2 = None
        self.rgbimg3 = None
        self.cloud1 = None
        self.cloud2 = None
        self.cloud3 = None
        self.rgb1 = None
        self.rgb2 = None
        self.rgb3 = None

    def pad_bottom_cloud(
        self, cloud, rgb_value=[255, 255, 255], for_visualization=False
    ):
        # generate 'fake' point cloud for area outside the bins
        r, g, b = rgb_value
        padding_more = 0.0
        if for_visualization:
            x = np.arange(
                (self.center[0] - self.x_half * 2) * 1000,
                (self.center[0] + self.x_half * 2) * 1000,
                2,
            )
            y = np.arange(
                (self.center[1] - self.y_half * 2) * 1000,
                (self.center[1] + self.y_half * 2) * 1000,
                2,
            )
        else:
            x = np.arange(
                (self.center[0] - self.x_half) * 1000,
                (self.center[0] + self.x_half) * 1000,
                2,
            )
            y = np.arange(
                (self.center[1] - self.y_half) * 1000,
                (self.center[1] + self.y_half) * 1000,
                2,
            )
        xx, yy = np.meshgrid(x, y)
        xx = xx / 1000
        yy = yy / 1000
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        pts = np.concatenate(
            [
                xx,
                yy,
                np.ones_like(yy) * (self.z_table),
                np.ones_like(yy) * r,
                np.ones_like(yy) * g,
                np.ones_like(yy) * b,
                np.zeros_like(yy),
            ],
            1,
        )
        # pts = pts[np.logical_not(((pts[:, 0] < self.center[0] + self.x_half) * (pts[:, 0] > self.center[0] - self.x_half) *
        #                           (pts[:, 1] < self.center[1] + self.y_half) * (pts[:, 1] > self.center[1] - self.y_half)))]
        # pts = pts[np.logical_not(((pts[:, 1] < 0.239 + half_size) * (pts[:, 1] > 0.239 - half_size)) + ((pts[:, 1] < -0.21 + half_size) * (pts[:, 1] > -0.21 - half_size)))]
        cloud = np.concatenate([cloud, pts], axis=0)
        return cloud

    def get_fused_clip_cloud(self, instruction, image_cond=None):
        self.clear_cache()
        cloud1 = self.get_pointcloud_from_depth(1, instruction, image_cond=image_cond)
        cloud1 = self.transform_clip_cloud_to_base(cloud1, 1)
        cloud2 = self.get_pointcloud_from_depth(2, instruction, image_cond=image_cond)
        cloud2 = self.transform_clip_cloud_to_base(cloud2, 2)
        cloud3 = self.get_pointcloud_from_depth(3, instruction, image_cond=image_cond)
        cloud3 = self.transform_clip_cloud_to_base(cloud3, 3)
        cloud = np.concatenate([cloud1, cloud2, cloud3], axis=0)
        # cloud = np.concatenate([cloud1, cloud3], axis=0)

        cloud = self.cloud_preprocess(cloud)
        cloud1 = self.cloud_preprocess(cloud1)
        cloud2 = self.cloud_preprocess(cloud2)
        cloud3 = self.cloud_preprocess(cloud3)

        cloud = self.pad_bottom_cloud(cloud, rgb_value=self.rgb_value)
        cloud2 = self.pad_bottom_cloud(cloud2, rgb_value=self.rgb_value)

        return cloud, cloud1, cloud2, cloud3

    def transform(self, cloud, T, isPosition=True):
        """Apply the homogeneous transform T to the point cloud. Use isPosition=False if transforming unit vectors."""

        n = cloud.shape[0]
        cloud = cloud.T
        augment = np.ones((1, n)) if isPosition else np.zeros((1, n))
        cloud = np.concatenate((cloud, augment), axis=0)
        cloud = np.dot(T, cloud)
        cloud = cloud[0:3, :].T
        return cloud

    def interpolate(self, depth):
        """
        Fill nans in depth image
        """
        # a boolean array of (width, height) which False where there are missing values and True where there are valid (non-missing) values
        mask = np.logical_not(np.isnan(depth))
        # array of (number of points, 2) containing the x,y coordinates of the valid values only
        xx, yy = np.meshgrid(np.arange(depth.shape[1]), np.arange(depth.shape[0]))
        xym = np.vstack((np.ravel(xx[mask]), np.ravel(yy[mask]))).T

        # the valid values in the first, second, third color channel,  as 1D arrays (in the same order as their coordinates in xym)
        data0 = np.ravel(depth[:, :][mask])

        # three separate interpolators for the separate color channels
        interp0 = scipy.interpolate.NearestNDInterpolator(xym, data0)

        # interpolate the whole image, one color channel at a time
        result0 = interp0(np.ravel(xx), np.ravel(yy)).reshape(xx.shape)

        return result0

    def getFilteredPointCloud(self, cloud, rgb, manual_offset=None):
        # filter ws x
        x_cond = (cloud[:, 0] < self.center[0] + self.x_half) * (
            cloud[:, 0] > self.center[0] - self.x_half
        )
        cloud = cloud[x_cond]
        rgb = rgb[x_cond]
        # filter ws y
        y_cond = (cloud[:, 1] < self.center[1] + self.y_half) * (
            cloud[:, 1] > self.center[1] - self.y_half
        )
        cloud = cloud[y_cond]
        rgb = rgb[y_cond]
        # filter ws z
        z_cond = (cloud[:, 2] < self.center[2].max()) * (cloud[:, 2] > self.z_min)
        cloud = cloud[z_cond]
        rgb = rgb[z_cond]

        if manual_offset is not None:
            # actually shouldn't do this. We should do cam intrinsic calib
            cloud[:, 0] += manual_offset[0]
            cloud[:, 1] += manual_offset[1]
            cloud[:, 2] += manual_offset[2]

        return cloud, rgb

    def getTopDownProjectImg(self, cloud_rgbc, projection_height=1.0, proj_pos=None):
        """
        return orthographic projection depth img from self.cloud
        target_size: img coverage size in meters
        img_size: img pixel size
        gripper_pos: the pos of the camera
        return depth image
        """
        if proj_pos is None:
            proj_pos = [self.center[0], self.center[1], projection_height]
        target_size = np.max([self.x_half, self.y_half]) * 2
        img_size = np.max([self.img_height, self.img_width])

        cloud = np.copy(cloud_rgbc[:, :3])
        rgbc = np.copy(cloud_rgbc[:, 3:])

        view_matrix = transformation.euler_matrix(0, np.pi, 0).dot(np.eye(4))
        # view_matrix = np.eye(4)
        view_matrix[:3, 3] = [proj_pos[0], -proj_pos[1], proj_pos[2]]
        view_matrix = transformation.euler_matrix(0, 0, 0).dot(view_matrix)
        # view_matrix[:3, 3] = [self.center[0], -self.center[1], projection_height]
        # view_matrix = transformation.euler_matrix(0, 0, 0).dot(view_matrix)
        augment = np.ones((1, cloud.shape[0]))
        pts = np.concatenate((cloud.T, augment), axis=0)
        scale = 1.15
        projection_matrix = np.array(
            [
                [scale / (target_size / 2), 0, 0, 0],
                [0, scale / (target_size / 2), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        tran_world_pix = np.matmul(projection_matrix, view_matrix)
        pts = np.matmul(tran_world_pix, pts)
        # pts[1] = -pts[1]
        pts[0] = (pts[0] + 1) * img_size / 2
        pts[1] = (pts[1] + 1) * img_size / 2

        pts[0] = np.round_(pts[0])
        pts[1] = np.round_(pts[1])
        mask = (pts[0] >= 0) * (pts[0] < img_size) * (pts[1] > 0) * (pts[1] < img_size)
        pts = pts[:, mask]
        # dense pixel index
        mix_xy = pts[1].astype(int) * img_size + pts[0].astype(int)
        # lexsort point cloud first on dense pixel index, then on z value
        ind = np.lexsort(np.stack((pts[2], mix_xy)))
        # bin count the points that belongs to each pixel
        bincount = np.bincount(mix_xy)
        # cumulative sum of the bin count. the result indicates the cumulative sum of number of points for all previous pixels
        cumsum = np.cumsum(bincount)
        # rolling the cumsum gives the ind of the first point that belongs to each pixel.
        # because of the lexsort, the first point has the smallest z value
        cumsum = np.roll(cumsum, 1)
        cumsum[0] = bincount[0]
        cumsum[cumsum == np.roll(cumsum, -1)] = 0
        # pad for unobserved pixels
        cumsum = np.concatenate(
            (cumsum, -1 * np.ones(img_size * img_size - cumsum.shape[0]))
        ).astype(int)

        depth = pts[2][ind][cumsum]
        depth[cumsum == 0] = np.nan
        depth = depth.reshape(img_size, img_size)
        # fill nans
        depth = self.interpolate(depth)

        rgbc = rgbc.T
        rgbc = rgbc[:, mask].T
        rgbc = rgbc[ind][cumsum]
        rgb = rgbc[:, :3]
        clip = rgbc[:, 3:4]
        rgb[cumsum == 0] = np.array(self.rgb_value)
        clip[cumsum == 0] = np.nan
        rgbc = np.concatenate([rgb, clip], axis=-1)
        rgbc = rgbc.reshape(img_size, img_size, 4)
        clip = self.interpolate(rgbc[..., 3])
        rgbc = np.concatenate([rgbc[..., :3], clip[..., None]], axis=2)

        img_left, img_right = (
            np.array([-self.img_width // 2, self.img_width // 2]) + img_size // 2
        )
        img_top, img_down = (
            np.array([-self.img_height // 2, self.img_height // 2]) + img_size // 2
        )
        depth = depth[img_top:img_down, img_left:img_right]
        rgbc = rgbc[img_top:img_down, img_left:img_right]
        return depth, rgbc

    def _preProcessObs(self, obs, kernel_size=5):
        obs = scipy.ndimage.median_filter(obs, kernel_size)
        return obs

    def _preProcessRGBC(self, obs, kernel_size=3):
        # obs = obs[..., :3]
        if len(obs.shape) == 2:
            obs = obs.reshape(obs.shape[0], obs.shape[1], 1)
        obss = []
        for channel in range(obs.shape[-1]):
            a = scipy.ndimage.median_filter(obs[..., channel], kernel_size)
            obss.append(a)
        obs = np.stack(obss).transpose(1, 2, 0)
        return obs

    def getHeightmapReconstruct(self, cloud_rgbc, separate_cloud=None):
        # get img from camera
        depth, rgbc = self.getTopDownProjectImg(cloud_rgbc)

        depth = self._preProcessObs(depth)
        rgb = cv2.bilateralFilter(rgbc[..., :3].astype(np.uint8), 15, 40, 40)

        clip_feature = rgbc[..., 3:4]
        clip_feature = (clip_feature - clip_feature.min()) / (
            clip_feature.max() - clip_feature.min()
        )
        rgbc = np.concatenate([rgb, clip_feature], axis=2)

        if separate_cloud is not None:
            clip_feature = []
            for cloud in separate_cloud:
                _, rgbc_tmp = self.getTopDownProjectImg(cloud)
                clip_feature.append(rgbc_tmp[..., 3])
            clip_feature = np.stack(clip_feature).mean(0)
            clip_feature = (clip_feature - clip_feature.min()) / (
                clip_feature.max() - clip_feature.min()
            )
            rgbc_average = np.concatenate([rgb, clip_feature[..., None]], axis=2)
        else:
            rgbc_average = None
        return depth, rgbc, rgbc_average

    def getClipObs(self, instruction, parsing=False, image_cond_2=None):
        if parsing:
            pick, place = parse_instruction(instruction, self.output_name)
            print(f"pick:{pick}, place:{place}")

            if image_cond_2 is not None:
                image_cond_pick, image_cond_place = image_cond_2[0], image_cond_2[1]
            else:
                image_cond_pick, image_cond_place = None, None

            cloud, cloud1, cloud2, cloud3 = self.get_fused_clip_cloud(
                pick, image_cond_pick
            )
            depth, rgbc, rgbc_average = self.getHeightmapReconstruct(
                cloud, [cloud1, cloud2, cloud3]
            )
            rgbc_average = self._preProcessRGBC(rgbc_average)
            clip_feature_pick = rgbc_average[..., 3]

            cloud, cloud1, cloud2, cloud3 = self.get_fused_clip_cloud(
                place, image_cond_place
            )
            depth, rgbc, rgbc_average = self.getHeightmapReconstruct(
                cloud, [cloud1, cloud2, cloud3]
            )
            rgbc_average = self._preProcessRGBC(rgbc_average)
            clip_feature_place = rgbc_average[..., 3]

            rgb = rgbc_average[..., :3]
        else:
            cloud, cloud1, cloud2, cloud3 = self.get_fused_clip_cloud(instruction)
            depth, rgbc, rgbc_average = self.getHeightmapReconstruct(
                cloud, [cloud1, cloud2, cloud3]
            )
            rgbc_average = self._preProcessRGBC(rgbc_average)
            clip_feature = rgbc_average[..., 3]
            rgb = rgbc_average[..., :3]
            clip_feature_pick = clip_feature
            clip_feature_place = clip_feature
        return depth, rgb, clip_feature_pick, clip_feature_place


if __name__ == "__main__":
    # datapath = './data/pick-part-in-brown-box.npy'
    file_path = "./data"
    file_list = [
        # 'pick-part-in-brown-box-demo1.npy',
        #  'pick-part-in-brown-box-demo2.npy',
        #  'pick-part-in-brown-box-demo3.npy',
        #  'demo_pentagon_in_blue_bowl.npy',
        # 'demo_greenmug_pp_new.npy',
        "pyramid_demo1.npy",
        "pyramid_demo2.npy",
        "pyramid_demo3.npy",
        "pyramid_demo4.npy",
        # 'demo5.npy',
        # 'pick-block-in-bowl-demo1.npy',
        # 'pick-block-in-bowl-demo2.npy',
        #  'pick-letter-on-color-plates-demo1.npy',
        #  'pick-letter-on-color-plates-demo2.npy',
        #  'pick-letter-on-color-plates-demo3.npy',
        #  'pick-letter-on-color-plates-demo4.npy',
        #  'pick-letter-on-color-plates-demo5.npy',
    ]
    # output_name = 'pick-part-in-brown-box-processed.npy'
    output_name = "pyramid-processed.npy"

    processor = real_data_processor(file_path, file_list, output_name)
    demos = processor.process_all_and_save()
    # processor.append_process_and_save('./data/pick-part-in-brown-box-processed-3.npy')
