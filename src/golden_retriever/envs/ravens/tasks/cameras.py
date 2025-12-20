"""Camera configs."""

import numpy as np
import pybullet as p


class RealSenseD415:
    """Default configuration with 3 RealSense RGB-D cameras."""

    # Mimic RealSense D415 RGB-D camera parameters.
    image_size = (480, 640)
    intrinsics = (450.0, 0, 320.0, 0, 450.0, 240.0, 0, 0, 1)

    # Set default camera poses.
    front_position = (1.0, 0, 0.75)
    front_rotation = (np.pi / 4, np.pi, -np.pi / 2)
    front_rotation = p.getQuaternionFromEuler(front_rotation)
    left_position = (0, 0.5, 0.75)
    left_rotation = (np.pi / 4.5, np.pi, np.pi / 4)
    left_rotation = p.getQuaternionFromEuler(left_rotation)
    right_position = (0, -0.5, 0.75)
    right_rotation = (np.pi / 4.5, np.pi, 3 * np.pi / 4)
    right_rotation = p.getQuaternionFromEuler(right_rotation)
    # Adjust position and rotation for top-down view.
    topdown_position = (
        0.5,
        0,
        0.71,
    )  # Position directly above the center of the scene at a height of 10 units.
    topdown_rotation = (
        0,
        np.pi,
        -np.pi / 2,
    )  # Euler angles for camera to look straight down.
    topdown_rotation = p.getQuaternionFromEuler(
        topdown_rotation
    )  # Convert Euler angles to a quaternion for the rotation.
    topdown_image_size = (320, 640)
    topdown_intrinsics = (450.0, 0, 320.0, 0, 450.0, 160.0, 0, 0, 1)
    # Default camera configs.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": front_position,
            "rotation": front_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": left_position,
            "rotation": left_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": right_position,
            "rotation": right_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
        {
            "image_size": topdown_image_size,
            "intrinsics": topdown_intrinsics,
            "position": topdown_position,
            "rotation": topdown_rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        },
    ]


class Oracle:
    """Top-down noiseless image used only by the oracle demonstrator."""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (63e4, 0, 320.0, 0, 63e4, 240.0, 0, 0, 1)
    position = (0.5, 0, 1000.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, -np.pi / 2))

    # Camera config.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": position,
            "rotation": rotation,
            "zrange": (999.7, 1001.0),
            "noise": False,
        }
    ]


class RS200Gazebo:
    """Gazebo Camera"""

    # Near-orthographic projection.
    image_size = (480, 640)
    intrinsics = (
        554.3826904296875,
        0.0,
        320.0,
        0.0,
        554.3826904296875,
        240.0,
        0.0,
        0.0,
        1.0,
    )
    position = (0.5, 0, 1.0)
    rotation = p.getQuaternionFromEuler((0, np.pi, np.pi / 2))

    # Camera config.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": position,
            "rotation": rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        }
    ]


class KinectFranka:
    """Kinect Franka Camera"""

    # Near-orthographic projection.
    image_size = (424, 512)
    intrinsics = (
        365.57489013671875,
        0.0,
        257.5205078125,
        0.0,
        365.57489013671875,
        205.26710510253906,
        0.0,
        0.0,
        1.0,
    )
    position = (1.082, -0.041, 1.027)
    rotation = p.getQuaternionFromEuler((-2.611, 0.010, 1.553))

    # Camera config.
    CONFIG = [
        {
            "image_size": image_size,
            "intrinsics": intrinsics,
            "position": position,
            "rotation": rotation,
            "zrange": (0.01, 10.0),
            "noise": False,
        }
    ]
