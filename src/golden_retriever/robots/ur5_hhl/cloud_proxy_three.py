# Adapted from Mingxi, Haojie

import time

import cv2
import numpy as np
import open3d
import scipy
import skimage.transform

from .clip_preprocess import CLIPFeatureProcessor
from .utils import transformation


def setup_ros_dependencies():
    global rospy, Image, CameraInfo, point_cloud2, PointCloud2, ros_numpy, tf2_ros, tf
    import ros_numpy
    import rospy
    import tf
    import tf2_ros
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import CameraInfo, Image, PointCloud2

    # from cv_bridge import CvBridge


class CloudProxy:
    def __init__(self, block_clip_processor=False, kernel_size=60, stride=20):
        # TODO Note: separate ROS dependencies
        setup_ros_dependencies()

        self.topic1 = "/cam_1/depth/color/points"
        self.topic2 = "/depth_to_rgb/points"
        self.topic3 = "/cam_3/depth/color/points"

        self.sub1 = rospy.Subscriber(
            self.topic1, PointCloud2, self.callbackCloud1, queue_size=1
        )
        self.msg1 = None
        self.sub2 = rospy.Subscriber(
            self.topic2, PointCloud2, self.callbackCloud2, queue_size=1
        )
        self.msg2 = None
        self.sub3 = rospy.Subscriber(
            self.topic3, PointCloud2, self.callbackCloud3, queue_size=1
        )
        self.msg3 = None
        self.image = None
        self.cloud1 = None
        self.cloud2 = None
        self.cloud3 = None
        self.rgb1 = None
        self.rgb2 = None
        self.rgb3 = None
        self.allow_color_cloud = True

        self.topic1_depth = "/cam_1/aligned_depth_to_color/image_raw"
        self.topic2_depth = "/depth_to_rgb/image_raw"
        self.topic3_depth = "/cam_3/aligned_depth_to_color/image_raw"
        self.sub1_depth = rospy.Subscriber(
            self.topic1_depth, Image, self.callbackDepth1, queue_size=1
        )
        self.depth1 = None
        self.sub2_depth = rospy.Subscriber(
            self.topic2_depth, Image, self.callbackDepth2, queue_size=1
        )
        self.depth2 = None
        self.sub3_depth = rospy.Subscriber(
            self.topic3_depth, Image, self.callbackDepth3, queue_size=1
        )
        self.depth3 = None

        self.topic1_info = "/cam_1/aligned_depth_to_color/camera_info"
        self.topic2_info = "/depth_to_rgb/camera_info"
        self.topic3_info = "/cam_3/aligned_depth_to_color/camera_info"
        self.sub1_info = rospy.Subscriber(
            self.topic1_info, CameraInfo, self.callbackInfo1, queue_size=1
        )
        self.info1 = None
        self.sub2_info = rospy.Subscriber(
            self.topic2_info, CameraInfo, self.callbackInfo2, queue_size=1
        )
        self.info2 = None
        self.sub3_info = rospy.Subscriber(
            self.topic3_info, CameraInfo, self.callbackInfo3, queue_size=1
        )
        self.info3 = None

        self.topic1_rgb = "/cam_1/color/image_raw"
        self.topic2_rgb = "/rgb/image_raw"
        self.topic3_rgb = "/cam_3/color/image_raw"
        self.sub1_rgb = rospy.Subscriber(
            self.topic1_rgb, Image, self.callbackRGB1, queue_size=1
        )
        self.rgbimg1 = None
        self.sub2_rgb = rospy.Subscriber(
            self.topic2_rgb, Image, self.callbackRGB2, queue_size=1
        )
        self.rgbimg2 = None
        self.sub3_rgb = rospy.Subscriber(
            self.topic3_rgb, Image, self.callbackRGB3, queue_size=1
        )
        self.rgbimg3 = None

        self.rgb1_frame = "cam_1_color_optical_frame"
        self.rgb2_frame = "rgb_camera_link"
        self.rgb3_frame = "cam_3_color_optical_frame"

        # RT base_link
        self.workspace = np.array([[0.22, 0.65], [-0.2, 0.37], [-0.3, 0.2]])
        self.center = self.workspace.mean(-1)
        self.x_size = self.workspace[0].max() - self.workspace[0].min()
        self.x_half = self.x_size / 2
        self.y_size = self.workspace[1].max() - self.workspace[1].min()
        self.y_half = self.y_size / 2
        self.z_min = self.workspace[2].min()

        self.tfBuffer = tf2_ros.Buffer()
        self.tfListener = tf2_ros.TransformListener(self.tfBuffer)
        self.base_frame = "base_link"

        if not block_clip_processor:
            self.clip_processor = CLIPFeatureProcessor()
        self.kernel_size = kernel_size
        self.stride = stride

        self.img_width = 240
        self.img_height = 320
        self.z_table = -0.173
        self.proj_pose = None
        self.rgb_value = [189, 152, 130]

        self.extrinsic2 = self.getCamExtrinsic(2)
        self.table_to_topdown_height = self.extrinsic2[2, -1] + np.abs(self.z_table)

    def callbackCloud1(self, msg):
        self.msg1 = msg
        cloudTime = self.msg1.header.stamp
        cloudFrame = self.msg1.header.frame_id
        cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(self.msg1)
        cloud, rgb = self.get_xyzrgb_points(cloud_arr, remove_nans=True)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        T = self.lookupTransform(cloudFrame, self.base_frame, rospy.Time(0))
        cloud = self.transform(cloud, T)
        self.cloud1 = cloud
        self.rgb1 = rgb

    def callbackCloud2(self, msg):
        self.msg2 = msg
        cloudTime = self.msg2.header.stamp
        cloudFrame = self.msg2.header.frame_id
        cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(self.msg2)
        cloud, rgb = self.get_xyzrgb_points(cloud_arr, remove_nans=True)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        T = self.lookupTransform(cloudFrame, self.base_frame, rospy.Time(0))
        cloud = self.transform(cloud, T)
        self.cloud2 = cloud
        self.rgb2 = rgb

    def callbackCloud3(self, msg):
        self.msg3 = msg
        cloudTime = self.msg3.header.stamp
        cloudFrame = self.msg3.header.frame_id
        cloud_arr = ros_numpy.point_cloud2.pointcloud2_to_array(self.msg3)
        cloud, rgb = self.get_xyzrgb_points(cloud_arr, remove_nans=True)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        T = self.lookupTransform(cloudFrame, self.base_frame, rospy.Time(0))
        cloud = self.transform(cloud, T)
        self.cloud3 = cloud
        self.rgb3 = rgb

    def callbackInfo1(self, msg):
        self.info1 = np.array(msg.K).reshape(3, 3)

    def callbackInfo2(self, msg):
        self.info2 = np.array(msg.K).reshape(3, 3)

    def callbackInfo3(self, msg):
        self.info3 = np.array(msg.K).reshape(3, 3)

    def callbackDepth1(self, msg):
        self.depth1 = ros_numpy.numpify(msg)

    def callbackDepth2(self, msg):
        self.depth2 = ros_numpy.numpify(msg)

    def callbackDepth3(self, msg):
        self.depth3 = ros_numpy.numpify(msg)

    def callbackRGB1(self, msg):
        self.rgbimg1 = ros_numpy.numpify(msg)

    def callbackRGB2(self, msg):
        self.rgbimg2 = ros_numpy.numpify(msg)

    def callbackRGB3(self, msg):
        self.rgbimg3 = ros_numpy.numpify(msg)

    def getDepthImage(self, cam_id, iteration=10):
        images = []
        for _ in range(iteration):
            if cam_id == 1:
                while self.depth1 is None:
                    rospy.sleep(0.01)
                images.append(self.depth1 / 1000)
            if cam_id == 2:
                while self.depth2 is None:
                    rospy.sleep(0.01)
                images.append(self.depth2)
            if cam_id == 3:
                while self.depth3 is None:
                    rospy.sleep(0.01)
                images.append(self.depth3 / 1000)
        image = np.median(images, axis=0)
        # self.image = self.image[240-100:240+100, 320-100:320+100]
        # self.image[np.isnan(self.image)] = 0
        # self.image = -self.image
        # self.image -= self.image.min()
        return image

    def getRGBImage(self, cam_id):
        if cam_id == 1:
            while self.rgbimg1 is None:
                rospy.sleep(0.01)
            return self.rgbimg1
        if cam_id == 2:
            while self.rgbimg2 is None:
                rospy.sleep(0.01)
            return cv2.cvtColor(self.rgbimg2, cv2.COLOR_BGRA2RGB)  # only for azure
        if cam_id == 3:
            while self.rgbimg3 is None:
                rospy.sleep(0.01)
            return self.rgbimg3

    def get_pointcloud_from_depth(self, cam_id, instruction, visualize=False):
        depth = self.getDepthImage(cam_id)
        rgb = self.getRGBImage(cam_id)
        kernel_size = self.kernel_size
        stride = self.stride

        clip_feature, _ = self.clip_processor.get_clip_feature(
            rgb, instruction, kernel_size=kernel_size, stride=stride
        )
        if cam_id == 1:
            while self.info1 is None:
                rospy.sleep(0.01)
            intrinsics = self.info1
        if cam_id == 2:
            while self.info2 is None:
                rospy.sleep(0.01)
            intrinsics = self.info2
        if cam_id == 3:
            while self.info3 is None:
                rospy.sleep(0.01)
            intrinsics = self.info3

        height, width = depth.shape
        xlin = np.linspace(0, width - 1, width)
        ylin = np.linspace(0, height - 1, height)
        px, py = np.meshgrid(xlin, ylin)
        px = (px - intrinsics[0, 2]) * (depth / intrinsics[0, 0])
        py = (py - intrinsics[1, 2]) * (depth / intrinsics[1, 1])

        points = np.float32(
            [px, py, depth, rgb[..., 0], rgb[..., 1], rgb[..., 2], clip_feature[..., 0]]
        ).transpose(1, 2, 0)
        cloud = points.reshape(-1, 7)
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
            rgb_frame = self.rgb1_frame
        if cam_id == 2:
            rgb_frame = self.rgb2_frame
            self.proj_pose = self.lookupTransform(
                rgb_frame, self.base_frame, rospy.Time(0)
            )
        if cam_id == 3:
            rgb_frame = self.rgb3_frame
        T = self.lookupTransform(rgb_frame, self.base_frame, rospy.Time(0))
        cloud_RT_base = self.transform(cloud[:, :3], T)
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

    def pad_bottom_cloud(self, cloud, rgb_value=[255, 255, 255]):
        # generate 'fake' point cloud for area outside the bins
        r, g, b = rgb_value
        padding_more = 0.0
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

    def get_fused_clip_cloud(self, instruction):
        self.clear_cache()
        cloud1 = self.get_pointcloud_from_depth(1, instruction)
        cloud1 = self.transform_clip_cloud_to_base(cloud1, 1)
        cloud2 = self.get_pointcloud_from_depth(2, instruction)
        cloud2 = self.transform_clip_cloud_to_base(cloud2, 2)
        cloud3 = self.get_pointcloud_from_depth(3, instruction)
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

    def get_fused_cloud(self, instruction):
        self.clear_cache()
        cloud1 = self.get_pointcloud_from_depth(1)
        cloud1 = self.transform_clip_cloud_to_base(cloud1, 1)
        cloud2 = self.get_pointcloud_from_depth(2)
        cloud2 = self.transform_clip_cloud_to_base(cloud2, 2)
        cloud3 = self.get_pointcloud_from_depth(3)
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

    def get_xyzrgb_points(self, cloud_array, remove_nans=True, dtype=np.float):
        """Pulls out x, y, and z columns from the cloud recordarray, and returns
        a 3xN matrix.
        """
        # remove crap points
        if remove_nans:
            mask = (
                np.isfinite(cloud_array["x"])
                & np.isfinite(cloud_array["y"])
                & np.isfinite(cloud_array["z"])
            )
            cloud_array = cloud_array[mask]

        # pull out x, y, and z values
        xyz = np.zeros(cloud_array.shape + (3,), dtype=dtype)
        xyz[..., 0] = cloud_array["x"]
        xyz[..., 1] = cloud_array["y"]
        xyz[..., 2] = cloud_array["z"]

        rgb_arr = ros_numpy.point_cloud2.split_rgb_field(cloud_array)
        rbg = np.zeros(cloud_array.shape + (3,), dtype=dtype)
        rbg[..., 0] = rgb_arr["r"]
        rbg[..., 1] = rgb_arr["g"]
        rbg[..., 2] = rgb_arr["b"]

        return xyz, rbg

    def transform(self, cloud, T, isPosition=True):
        """Apply the homogeneous transform T to the point cloud. Use isPosition=False if transforming unit vectors."""

        n = cloud.shape[0]
        cloud = cloud.T
        augment = np.ones((1, n)) if isPosition else np.zeros((1, n))
        cloud = np.concatenate((cloud, augment), axis=0)
        cloud = np.dot(T, cloud)
        cloud = cloud[0:3, :].T
        return cloud

    def lookupTransform(self, fromFrame, toFrame, lookupTime=None):
        """
        Lookup a transform in the TF tree.
        :param fromFrame: the frame from which the transform is calculated
        :type fromFrame: string
        :param toFrame: the frame to which the transform is calculated
        :type toFrame: string
        :return: transformation matrix from fromFrame to toFrame
        :rtype: 4x4 np.array
        """

        # Note: workaround for not importing rospy in file level
        if lookupTime is None:
            lookupTime = rospy.Time(0)

        transformMsg = self.tfBuffer.lookup_transform(
            toFrame, fromFrame, lookupTime, rospy.Duration(1.0)
        )
        translation = transformMsg.transform.translation
        pos = [translation.x, translation.y, translation.z]
        rotation = transformMsg.transform.rotation
        quat = [rotation.x, rotation.y, rotation.z, rotation.w]
        T = tf.transformations.quaternion_matrix(quat)
        T[0:3, 3] = pos
        return T

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

    def getFusedPointCloud(self):
        """
        get new point cloud, set self.cloud
        """
        start_time = time.time()
        while (
            self.cloud1 is None
            or self.cloud2 is None
            or self.cloud3 is None
            or self.rgb1 is None
            or self.rgb2 is None
            or self.rgb3 is None
        ):
            rospy.sleep(0.1)

        cloud1, rgb1 = self.getFilteredPointCloud(self.cloud1, self.rgb1)
        cloud2, rgb2 = self.getFilteredPointCloud(self.cloud2, self.rgb2)
        cloud3, rgb3 = self.getFilteredPointCloud(self.cloud3, self.rgb3)
        # cloud3, rgb3 = self.getFilteredPointCloud(self.cloud3, self.rgb3, manual_offset=[0.022, 0.005, 0.05])

        cloud = np.concatenate((cloud1, cloud2, cloud3))
        rgb = np.concatenate((rgb1, rgb2, rgb3))

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(cloud)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=50, std_ratio=2.0)
        cloud = np.asarray(cl.points)
        rgb = rgb[ind]

        # generate 'fake' point cloud for area outside the bins
        padding_more = 0.0
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
        xx, yy = np.meshgrid(x, y)
        xx = xx / 1000
        yy = yy / 1000
        xx = xx.reshape(-1, 1)
        yy = yy.reshape(-1, 1)
        pts = np.concatenate([xx, yy, np.ones_like(yy) * (self.z_min)], 1)
        # pts = pts[np.logical_not(((pts[:, 0] < self.center[0] + self.x_half) * (pts[:, 0] > self.center[0] - self.x_half) *
        #                           (pts[:, 1] < self.center[1] + self.y_half) * (pts[:, 1] > self.center[1] - self.y_half)))]
        # pts = pts[np.logical_not(((pts[:, 1] < 0.239 + half_size) * (pts[:, 1] > 0.239 - half_size)) + ((pts[:, 1] < -0.21 + half_size) * (pts[:, 1] > -0.21 - half_size)))]
        cloud = np.concatenate([cloud, pts])
        self.cloud = cloud

        if self.allow_color_cloud:
            # predefined_color = [170,166,159]
            predefined_color = [255, 255, 255]
            padding_rgb = np.ones([pts.shape[0], 3]) * np.array(predefined_color)
            rgb = np.concatenate([rgb, padding_rgb])
            cloud_color = np.concatenate([cloud, rgb], axis=-1)
            self.cloud_color = cloud_color
        else:
            cloud_color = None
            self.cloud_color = cloud_color

        return cloud1, cloud2, cloud3, cloud, cloud_color

    def getProjectImg(self, target_size, img_size, gripper_pos=(-0.5, 0, 0.1)):
        """
        return orthographic projection depth img from self.cloud
        target_size: img coverage size in meters
        img_size: img pixel size
        gripper_pos: the pos of the camera
        return depth image
        """
        if self.cloud is None:
            self.getNewPointCloud(gripper_pos)
        cloud = np.copy(self.cloud)
        cloud = cloud[(cloud[:, 2] < max(gripper_pos[2], self.z_min + 0.05))]
        view_matrix = transformation.euler_matrix(0, np.pi, 0).dot(np.eye(4))
        # view_matrix = np.eye(4)
        view_matrix[:3, 3] = [gripper_pos[0], -gripper_pos[1], gripper_pos[2]]
        view_matrix = transformation.euler_matrix(0, 0, -np.pi / 2).dot(view_matrix)
        augment = np.ones((1, cloud.shape[0]))
        pts = np.concatenate((cloud.T, augment), axis=0)
        projection_matrix = np.array(
            [
                [1 / (target_size / 2), 0, 0, 0],
                [0, 1 / (target_size / 2), 0, 0],
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
        # mask = np.isnan(depth)
        # depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])
        # imputer = SimpleImputer(missing_values=np.nan, strategy='median')
        # imputer_depth = imputer.fit_transform(depth)
        # if imputer_depth.shape != depth.shape:
        #     mask = np.isnan(depth)
        #     depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])
        # else:
        #     depth = imputer_depth
        return depth

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

    def getClipObs(self, instruction, parsing=False, task_name=None):
        def parse_instruction(instruction):
            if instruction.count(" and ") == 2:
                # temp solution for pyramid in multitask training

                pick, place = instruction.split(" on ")
                pick = " ".join(pick.split(" ")[1:-2])
                place = " ".join(place.split(" ")[:])

            # if "pyramid" in task_name:
            #     pick, place = instruction.split(' on ')
            #     pick = " ".join(pick.split(' ')[1:-2])
            #     place = " ".join(place.split(' ')[:])
            else:
                pick, place = instruction.split(" and ")
                pick = " ".join(pick.split(" ")[1:])
                place = " ".join(place.split(" ")[2:])

            return pick, place

        if parsing:
            pick, place = parse_instruction(instruction)
            print(f"pick:{pick}, place:{place}")

            cloud, cloud1, cloud2, cloud3 = self.get_fused_clip_cloud(pick)
            depth, _, rgbc_average = self.getHeightmapReconstruct(
                cloud, [cloud1, cloud2, cloud3]
            )
            clip_feature_pick = rgbc_average[..., 3]
            rgbc_average = self._preProcessRGBC(rgbc_average)

            cloud, cloud1, cloud2, cloud3 = self.get_fused_clip_cloud(place)
            depth, _, rgbc_average = self.getHeightmapReconstruct(
                cloud, [cloud1, cloud2, cloud3]
            )
            clip_feature_place = rgbc_average[..., 3]
            rgbc_average = self._preProcessRGBC(rgbc_average)

            rgb = rgbc_average[..., :3]
        else:
            pick, place = parse_instruction(instruction)
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

    def getCamIntrinsic(self, cam_id):
        if cam_id == 1:
            while self.info1 is None:
                rospy.sleep(0.01)
            intrinsics = self.info1
        if cam_id == 2:
            while self.info2 is None:
                rospy.sleep(0.01)
            intrinsics = self.info2
        if cam_id == 3:
            while self.info3 is None:
                rospy.sleep(0.01)
            intrinsics = self.info3
        return intrinsics

    def getCamExtrinsic(self, cam_id):
        if cam_id == 1:
            rgb_frame = self.rgb1_frame
            extrinsic = self.lookupTransform(rgb_frame, self.base_frame, rospy.Time(0))
        if cam_id == 2:
            rgb_frame = self.rgb2_frame
            extrinsic = self.lookupTransform(rgb_frame, self.base_frame, rospy.Time(0))
        if cam_id == 3:
            rgb_frame = self.rgb3_frame
            extrinsic = self.lookupTransform(rgb_frame, self.base_frame, rospy.Time(0))
        return extrinsic

    def get_multi_obs(self):
        self.clear_cache()
        depth = self.getDepthImage(1)
        rgb = self.getRGBImage(1)
        rgbd1 = np.concatenate([rgb, depth[..., None]], axis=-1)
        intrinsic1 = self.getCamIntrinsic(1)
        extrinsic1 = self.getCamExtrinsic(1)

        depth = self.getDepthImage(2)
        rgb = self.getRGBImage(2)
        rgbd2 = np.concatenate([rgb, depth[..., None]], axis=-1)
        intrinsic2 = self.getCamIntrinsic(2)
        extrinsic2 = self.getCamExtrinsic(2)

        depth = self.getDepthImage(3)
        rgb = self.getRGBImage(3)
        rgbd3 = np.concatenate([rgb, depth[..., None]], axis=-1)
        intrinsic3 = self.getCamIntrinsic(3)
        extrinsic3 = self.getCamExtrinsic(3)
        return (
            rgbd1,
            rgbd2,
            rgbd3,
            intrinsic1,
            intrinsic2,
            intrinsic3,
            extrinsic1,
            extrinsic2,
            extrinsic3,
        )

    def getObs(self, instruction, block_clip_processor=False, task_name=None):
        if not block_clip_processor:
            _, _, clip_feature_pick, clip_feature_place = self.getClipObs(
                instruction, parsing=True, task_name=task_name
            )
        else:
            clip_feature_pick, clip_feature_place = None, None

        depth = np.rot90(self.getDepthImage(2)[190:480, 515:725], 2)
        rgb = np.rot90(self.getRGBImage(2)[190:480, 515:725], 2)
        depth = skimage.transform.resize(depth, (self.img_height, self.img_width))
        rgb = skimage.transform.resize(
            rgb.astype(float), (self.img_height, self.img_width, 3)
        )

        return (
            np.clip(self.table_to_topdown_height - depth, 0, 0.3),
            rgb,
            clip_feature_pick,
            clip_feature_place,
        )


def main():
    rospy.init_node("test")
    cloudProxy = CloudProxy()
    img_ = None
    while True:
        cloudProxy.cloud = None
        # img = cloudProxy.get_pointcloud_from_depth(1, "yellow block")
        instruction = "letter X block"
        cloud, cloud1, cloud2, cloud3 = cloudProxy.get_fused_clip_cloud(instruction)
        # cloud1, cloud2, cloud3, cloud, cloud_color = cloudProxy.getFusedPointCloud()
        visualize = True
        if visualize:
            pcd = open3d.geometry.PointCloud()
            pcd.points = open3d.utility.Vector3dVector(cloud[:, :3])
            clip_cloud = np.copy(cloud[:, 6:7])
            clip_cloud = (clip_cloud - clip_cloud.min()) / (
                clip_cloud.max() - clip_cloud.min()
            )
            pcd.colors = open3d.utility.Vector3dVector(cloud[:, 3:6] * clip_cloud / 255)
            open3d.visualization.draw_geometries([pcd])
        depth, rgbc, rgbc_average = cloudProxy.getHeightmapReconstruct(
            cloud, [cloud1, cloud2, cloud3]
        )
        depth, rgbc, rgbc_average = cloudProxy.getHeightmapReconstruct(cloud2)
        rgbc = cloudProxy._preProcessRGBC(rgbc)


if __name__ == "__main__":
    main()
