# Adapted from Mingxi, Haojie

import matplotlib.pyplot as plt
import numpy as np

import src.robots.ur5_hhl.utils.demo_util as demo_util

WS_CENTER = [320, 628]
OBS_X_OFFSET = 280
OBS_Y_OFFSET = 140


def setup_ros_dependencies():
    global ros_numpy, rospy, Image
    import ros_numpy
    import rospy
    from sensor_msgs.msg import Image


class ImgProxy:
    def __init__(self):
        # TODO Note: separate ROS dependencies
        setup_ros_dependencies()

        self.topic_depth = "/rgb/image_rect_color"
        self.sub_depth = rospy.Subscriber(
            self.topic_depth, Image, self.callbackImage, queue_size=1
        )
        self.msg = None
        self.image = None
        self.has_image = True

    def callbackImage(self, msg):
        if not self.has_image:
            self.msg = msg
            self.has_image = True

    def getImage(self, iteration=10):
        images = []
        for _ in range(iteration):
            self.has_image = False
            while not self.has_image:
                rospy.sleep(0.01)
            img = ros_numpy.numpify(self.msg)
            # mask = np.isnan(img)
            # img[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), img[~mask])
            images.append(img)
        self.image = np.median(images, axis=0)
        # self.image = self.image[240-100:240+100, 320-100:320+100]
        # self.image[np.isnan(self.image)] = 0
        # self.image = -self.image
        # self.image -= self.image.min()
        return self.image

    def getTableImage(self):
        obs = self.getImage()
        obs = obs[
            WS_CENTER[0] - OBS_X_OFFSET : WS_CENTER[0] + OBS_X_OFFSET,
            WS_CENTER[1] - OBS_Y_OFFSET : WS_CENTER[1] + OBS_Y_OFFSET,
            :3,
        ]
        obs = obs[:, :, ::-1].astype(int)
        obs = np.transpose(obs, (1, 0, 2))
        obs = np.flip(obs, 0)
        obs[np.isnan(obs)] = 0
        return obs

    def getObsImage(self):
        obs = self.getTableImage()
        obs = obs[
            demo_util.UPPER_PIX : demo_util.DOWN_PIX,
            demo_util.LEFT_PIX : demo_util.RIGHT_PIX,
            :,
        ]
        return obs


class DepthProxy:
    def __init__(self):
        self.topic = "/depth_to_rgb/image_raw"
        self.sub = rospy.Subscriber(self.topic, Image, self.callbackImage, queue_size=1)
        self.msg = None
        self.image = None
        self.has_image = True

    def callbackImage(self, msg):
        if not self.has_image:
            self.msg = msg
            self.has_image = True

    def getImage(self, iteration=10):
        images = []
        for _ in range(iteration):
            self.has_image = False
            while not self.has_image:
                rospy.sleep(0.01)
            img = ros_numpy.numpify(self.msg)
            # mask = np.isnan(img)
            # img[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), img[~mask])
            images.append(img)
        self.image = np.median(images, axis=0)
        # self.image = self.image[240-100:240+100, 320-100:320+100]
        # self.image[np.isnan(self.image)] = 0
        # self.image = -self.image
        # self.image -= self.image.min()
        return self.image

    def getTableImage(self):
        obs = self.getImage()
        obs = obs[
            WS_CENTER[0] - OBS_X_OFFSET : WS_CENTER[0] + OBS_X_OFFSET,
            WS_CENTER[1] - OBS_Y_OFFSET : WS_CENTER[1] + OBS_Y_OFFSET,
        ]
        obs = np.transpose(obs, (1, 0))
        obs = np.flip(obs, 0)
        obs[np.isnan(obs)] = 0
        return obs

    def getObsImage(self):
        obs = self.getTableImage()
        obs = obs[
            demo_util.UPPER_PIX : demo_util.DOWN_PIX,
            demo_util.LEFT_PIX : demo_util.RIGHT_PIX,
        ]
        return obs


def main():
    plt.style.use("grayscale")
    rospy.init_node("image_proxy")
    imgProxy = ImgProxy()
    depthProxy = DepthProxy()
    while True:
        obs = imgProxy.getImage()
        depth = depthProxy.getImage()
        obs = obs[320 - 280 : 320 + 280, 628 - 140 : 628 + 140, :3]
        obs = obs[:, :, ::-1].astype(int)
        obs = np.transpose(obs, (1, 0, 2))
        obs = np.flip(obs, 0)
        # obs = obs[:,:,::-1]
        obs[np.isnan(obs)] = 0
        # obs = skimage.transform.rotate(obs, 90)
        # obs = -obs
        # obs -= obs.min()
        # obs = skimage.transform.resize(obs, (90, 90))
        # obs = obs[100-10:100+10, 100-10:100+10]
        plt.imshow(obs)
        plt.colorbar()
        plt.show()
        print(1)


if __name__ == "__main__":
    main()
