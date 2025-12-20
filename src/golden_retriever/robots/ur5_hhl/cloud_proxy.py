# Adapted from Mingxi, Haojie

import matplotlib.pyplot as plt
import numpy as np
from skimage.restoration import inpaint
from skimage.transform import rotate


def setup_ros_dependencies():
    global rospy, ros_numpy, Image, PointCloud2
    import ros_numpy
    import rospy
    from sensor_msgs.msg import Image, PointCloud2


class CloudProxy:
    def __init__(self):
        # TODO Note: separate ROS dependencies
        setup_ros_dependencies()

        # self.topic = '/camera/depth/points'
        # self.topic = '/kinect2/hd/points'
        self.topic = "/depth_to_rgb/points"
        self.sub = rospy.Subscriber(
            self.topic, PointCloud2, self.callbackCloud, queue_size=1
        )
        self.msg = None
        self.image = None
        self.has_cloud = False

    def callbackCloud(self, msg):
        if not self.has_cloud:
            self.msg = msg
            self.has_cloud = True

    def getCloud(self):
        """
        get cloud in camera frame.
        :return: point cloud, cloud frame, cloud time
        """
        self.has_cloud = False
        while not self.has_cloud:
            rospy.sleep(0.005)
        cloudTime = self.msg.header.stamp
        cloudFrame = self.msg.header.frame_id
        pc = ros_numpy.numpify(self.msg)
        pc = ros_numpy.point_cloud2.split_rgb_field(pc)
        height = pc.shape[0]
        width = pc.shape[1]
        cloud = np.zeros((height * width, 6), dtype=np.float32)
        cloud[:, 0] = np.resize(pc["x"], height * width)
        cloud[:, 1] = np.resize(pc["y"], height * width)
        cloud[:, 2] = np.resize(pc["z"], height * width)
        cloud[:, 3] = np.resize(pc["r"], height * width)
        cloud[:, 4] = np.resize(pc["g"], height * width)
        cloud[:, 5] = np.resize(pc["b"], height * width)
        mask = np.logical_not(np.isnan(cloud).any(axis=1))
        cloud = cloud[mask]
        # print("Received Structure cloud with {} points.".format(cloud.shape[0]))
        return cloud

    def getProjectImg(self, target_size, img_size, return_rgb=False, return_mask=False):
        clouds = []
        # start_time = time.time()
        for i in range(2):
            clouds.append(self.getCloud())
        # print('get getCloud: ', time.time() - start_time)
        view_matrix = np.eye(4)
        view_matrix[:3, 3] = [-0.007, -0.013, 0]
        # augment = np.ones((1, cloud.shape[0]))
        # pts = np.concatenate((cloud.T, augment), axis=0)
        scale = 24.3 / 25
        projection_matrix = np.array(
            [
                [scale / (target_size / 2), 0, 0, 0],
                [0, scale / (target_size / 2), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        tran_world_pix = np.matmul(projection_matrix, view_matrix)
        depths, rgbs, masks = [], [], []
        # start_time = time.time()
        rotation = 89.2
        for cloud in clouds:
            pts = cloud.T
            pts[:2] = np.matmul(tran_world_pix[:2, :2], pts[:2])
            # pts[1] = -pts[1]
            # pts[0] = (pts[0] + 1) * img_size / 2 - 2
            pts[0] = (pts[0] + 1) * img_size / 2
            # pts[1] = (pts[1] + 1) * img_size / 2 + 20
            pts[1] = (pts[1] + 1) * img_size / 2 + 18

            pts[0] = np.round_(pts[0])
            pts[1] = np.round_(pts[1])
            mask = (
                (pts[0] >= 0) * (pts[0] < img_size) * (pts[1] > 0) * (pts[1] < img_size)
            )
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
            # depth[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), depth[~mask])
            # imputer = SimpleImputer(missing_values=np.nan, strategy='median')
            # depth = imputer.fit_transform(depth)
            depth = rotate(depth, rotation)
            # depth[mask] = 0
            assert depth.shape == (img_size, img_size)
            depths.append(depth)
            # masks.append(mask)
            if return_rgb:
                rgb = pts[3:][:, ind][:, cumsum]
                rgb = rgb.T.reshape(img_size, img_size, 3)
                rgb = rotate(rgb, rotation)
                rgbs.append(rgb)

        depth = np.nanmedian(depths, axis=0)
        mask = np.isnan(depth)
        depth = inpaint.inpaint_biharmonic(depth, mask)

        # print('return obs: ', time.time() - start_time)
        if return_rgb:
            rgb = np.nanmedian(rgbs, axis=0)
            rgb = inpaint.inpaint_biharmonic(rgb, mask, channel_axis=-1)
            rgb = rgb.transpose(2, 0, 1)
            rgb /= 255
            rgb -= 0.5
            if not return_mask:
                return depth, rgb
            else:
                # mask = np.nanmedian(masks, axis=0)
                return depth, rgb, mask
        else:
            return depth


def main():
    rospy.init_node("test")
    cloudProxy = CloudProxy()
    img_ = None
    while True:
        obs, rgb, mask = cloudProxy.getProjectImg(
            0.8, 128 * 2, return_rgb=True, return_mask=True
        )
        # plt.figure()
        # plt.imshow(mask)
        # plt.colorbar()
        plt.figure()
        plt.imshow(obs)
        plt.colorbar()
        plt.figure()
        img = rgb.transpose(1, 2, 0)
        img += 0.05
        img *= 2550
        plt.imshow(img.astype(int))
        if img_ is not None:
            plt.figure()
            plt.imshow((img - img_).sum(axis=-1))
            plt.colorbar()
        img_ = img
        # plt.imshow(rgb.astype(int))
        plt.show()


if __name__ == "__main__":
    main()
