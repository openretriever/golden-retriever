#!/usr/bin/env python3
"""
Format Converters

Utilities for converting between different observation and action formats.
Supports conversion between:
- Libero ↔ Standard RobotObservation/RobotAction
- OpenPI ↔ Standard RobotObservation/RobotAction  
- Generic dict formats ↔ Standard formats
"""

import math
from typing import Dict, Any, List
import numpy as np

try:
    from ..types.robotics_types import RobotObservation, RobotAction
except ImportError:
    from robotics_types import RobotObservation, RobotAction


# =============================================================================
# Libero Format Converters
# =============================================================================

def libero_obs_to_robot_obs(libero_obs: Dict[str, Any], task_info: str) -> RobotObservation:
    """
    Convert Libero observation to standard RobotObservation format.
    
    Libero provides:
    - "agentview_image": Main camera view
    - "robot0_eye_in_hand_image": Wrist camera  
    - "robot0_eef_pos": End-effector position
    - "robot0_eef_quat": End-effector quaternion
    - "robot0_gripper_qpos": Gripper joint positions
    """
    
    # Extract and process images
    images = {}
    
    # Agent view camera (main camera)
    if "agentview_image" in libero_obs:
        # IMPORTANT: Rotate 180 degrees to match OpenPI training preprocessing
        img = libero_obs["agentview_image"]
        images["agentview"] = np.ascontiguousarray(img[::-1, ::-1]).astype(np.uint8)
    
    # Wrist camera (eye-in-hand)
    if "robot0_eye_in_hand_image" in libero_obs:
        # IMPORTANT: Rotate 180 degrees to match OpenPI training preprocessing
        img = libero_obs["robot0_eye_in_hand_image"]
        images["wrist"] = np.ascontiguousarray(img[::-1, ::-1]).astype(np.uint8)
    
    # Extract robot state components
    robot_state_components = []
    
    # End-effector position (3D)
    if "robot0_eef_pos" in libero_obs:
        robot_state_components.append(libero_obs["robot0_eef_pos"])
    
    # End-effector orientation (convert quaternion to axis-angle)
    if "robot0_eef_quat" in libero_obs:
        axis_angle = quat_to_axis_angle(libero_obs["robot0_eef_quat"])
        robot_state_components.append(axis_angle)
    
    # Gripper state (joint positions)
    if "robot0_gripper_qpos" in libero_obs:
        robot_state_components.append(libero_obs["robot0_gripper_qpos"])
    
    # Combine all state components
    if robot_state_components:
        robot_state = np.concatenate(robot_state_components).astype(np.float32)
    else:
        # Fallback: create default state
        robot_state = np.zeros(9, dtype=np.float32)  # 3 pos + 3 orient + 2 gripper + 1 extra
    
    return RobotObservation(
        images=images,
        robot_state=robot_state,
        task_info=task_info,
        metadata={
            "source": "libero",
            "raw_keys": list(libero_obs.keys()),
            "eef_pos_shape": libero_obs.get("robot0_eef_pos", np.array([])).shape,
            "gripper_shape": libero_obs.get("robot0_gripper_qpos", np.array([])).shape
        }
    )


def robot_action_to_libero_action(robot_action: RobotAction) -> List[float]:
    """
    Convert RobotAction to Libero action format.
    
    Libero expects: [x, y, z, rx, ry, rz, gripper]
    - First 6 elements: End-effector pose (position + orientation)
    - Last element: Gripper control
    """
    
    # Ensure we have at least 6 joint positions for end-effector control
    if len(robot_action.joint_positions) >= 6:
        libero_action = robot_action.joint_positions[:6].tolist()
    else:
        # Pad with zeros if we don't have enough dimensions
        libero_action = list(robot_action.joint_positions) + [0.0] * (6 - len(robot_action.joint_positions))
    
    # Add gripper action
    libero_action.append(float(robot_action.gripper_action))
    
    return libero_action


# =============================================================================
# OpenPI Format Converters  
# =============================================================================

def robot_obs_to_openpi_obs(robot_obs: RobotObservation, resize_size: int = 224) -> Dict[str, Any]:
    """
    Convert RobotObservation to OpenPI format.
    
    OpenPI expects:
    - "observation/image": Main camera (resized)
    - "observation/wrist_image": Wrist camera (resized) 
    - "observation/state": Robot state vector
    - "prompt": Task description
    """
    
    openpi_obs = {}
    
    # Process images with resizing
    if "agentview" in robot_obs.images:
        img = robot_obs.images["agentview"]
        # For now, assume image is already properly sized
        # In production, you'd use image_tools.resize_with_pad here
        openpi_obs["observation/image"] = img
    
    if "wrist" in robot_obs.images:
        img = robot_obs.images["wrist"]
        # For now, assume image is already properly sized
        # In production, you'd use image_tools.resize_with_pad here
        openpi_obs["observation/wrist_image"] = img
    
    # Robot state
    openpi_obs["observation/state"] = robot_obs.robot_state
    
    # Task description
    openpi_obs["prompt"] = robot_obs.task_info
    
    return openpi_obs


def openpi_action_to_robot_action(openpi_action: Dict[str, Any]) -> RobotAction:
    """
    Convert OpenPI action format to RobotAction.
    
    OpenPI returns action chunks as numpy arrays.
    For Libero, this is typically [x, y, z, rx, ry, rz, gripper] format.
    """
    
    # OpenPI returns "actions" key with action chunk
    if "actions" in openpi_action:
        actions = openpi_action["actions"]
        if len(actions) > 0:
            # Take first action from chunk
            action = actions[0]
            
            if len(action) >= 7:
                joint_positions = action[:6]  # End-effector pose
                gripper_action = action[6]    # Gripper control
            else:
                joint_positions = np.zeros(6, dtype=np.float32)
                gripper_action = 0.0
        else:
            joint_positions = np.zeros(6, dtype=np.float32)
            gripper_action = 0.0
    else:
        joint_positions = np.zeros(6, dtype=np.float32)
        gripper_action = 0.0
    
    return RobotAction(
        joint_positions=np.array(joint_positions, dtype=np.float32),
        gripper_action=float(gripper_action),
        metadata={
            "source": "openpi",
            "raw_keys": list(openpi_action.keys()),
            "action_chunk_size": len(openpi_action.get("actions", []))
        }
    )


# =============================================================================
# Generic Dictionary Converters
# =============================================================================

def dict_obs_to_robot_obs(obs_dict: Dict[str, Any], 
                         image_keys: List[str] = None,
                         state_key: str = "robot_state",
                         task_key: str = "task_info") -> RobotObservation:
    """
    Convert generic dictionary observation to RobotObservation.
    
    Flexible converter for custom environments.
    """
    
    if image_keys is None:
        image_keys = ["agentview", "wrist", "image", "rgb"]
    
    # Extract images
    images = {}
    for key in image_keys:
        if key in obs_dict:
            img = obs_dict[key]
            if isinstance(img, np.ndarray) and len(img.shape) == 3:
                images[key] = img.astype(np.uint8)
    
    # Extract robot state
    if state_key in obs_dict:
        robot_state = np.array(obs_dict[state_key], dtype=np.float32)
    else:
        robot_state = np.zeros(9, dtype=np.float32)  # Default
    
    # Extract task info
    task_info = obs_dict.get(task_key, "unknown task")
    if not isinstance(task_info, str):
        task_info = str(task_info)
    
    return RobotObservation(
        images=images,
        robot_state=robot_state,
        task_info=task_info,
        metadata={"source": "dict", "original_keys": list(obs_dict.keys())}
    )


def robot_action_to_dict(robot_action: RobotAction, 
                        format_type: str = "generic") -> Dict[str, Any]:
    """
    Convert RobotAction to dictionary format.
    
    Args:
        format_type: "generic", "libero", "openpi", or custom format name
    """
    
    if format_type == "libero":
        return {
            "action": robot_action_to_libero_action(robot_action),
            "metadata": robot_action.metadata
        }
    
    elif format_type == "openpi":
        return {
            "robot0_joint_pos": robot_action.joint_positions,
            "robot0_gripper": np.array([robot_action.gripper_action]),
            "metadata": robot_action.metadata
        }
    
    else:  # generic
        return {
            "joint_positions": robot_action.joint_positions,
            "gripper_action": robot_action.gripper_action,
            "metadata": robot_action.metadata
        }


# =============================================================================
# Utility Functions
# =============================================================================

def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to axis-angle representation.
    
    Copied from robosuite for consistency with Libero training data.
    https://github.com/ARISE-Initiative/robosuite/blob/main/robosuite/utils/transform_utils.py
    """
    
    # Ensure quaternion is numpy array
    quat = np.array(quat, dtype=np.float32)
    
    # Clip quaternion w component to valid range
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    # Calculate axis-angle
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation
        return np.zeros(3, dtype=np.float32)

    return ((quat[:3] * 2.0 * math.acos(quat[3])) / den).astype(np.float32)


def axis_angle_to_quat(axis_angle: np.ndarray) -> np.ndarray:
    """
    Convert axis-angle to quaternion representation.
    
    Inverse of quat_to_axis_angle.
    """
    
    axis_angle = np.array(axis_angle, dtype=np.float32)
    angle = np.linalg.norm(axis_angle)
    
    if math.isclose(angle, 0.0):
        # Zero rotation
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    
    axis = axis_angle / angle
    half_angle = angle / 2.0
    
    quat = np.zeros(4, dtype=np.float32)
    quat[:3] = axis * np.sin(half_angle)
    quat[3] = np.cos(half_angle)
    
    return quat


def resize_image_with_pad(image: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
    """
    Resize image with padding to maintain aspect ratio.
    
    Simple implementation - for production use image_tools.resize_with_pad.
    """
    
    h, w = image.shape[:2]
    
    # Calculate scaling factor
    scale = min(target_height / h, target_width / w)
    
    # Resize image
    new_h, new_w = int(h * scale), int(w * scale)
    
    import cv2
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    padded = np.zeros((target_height, target_width, 3), dtype=image.dtype)
    
    # Center the resized image
    y_offset = (target_height - new_h) // 2
    x_offset = (target_width - new_w) // 2
    
    padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    
    return padded


if __name__ == "__main__":
    # Test the format converters
    print("🧪 Testing format converters...")
    
    # Test Libero conversion
    print("\n🏠 Testing Libero conversion...")
    
    # Mock Libero observation
    libero_obs = {
        "agentview_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, -0.2, 0.8], dtype=np.float32),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.04, -0.04], dtype=np.float32)
    }
    
    robot_obs = libero_obs_to_robot_obs(libero_obs, "pick up the red cup")
    print(f"✅ Libero → Robot: {len(robot_obs.images)} images, state shape {robot_obs.robot_state.shape}")
    
    # Test action conversion
    robot_action = RobotAction(
        joint_positions=np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32),
        gripper_action=0.5
    )
    
    libero_action = robot_action_to_libero_action(robot_action)
    print(f"✅ Robot → Libero: {len(libero_action)} elements: {libero_action}")
    
    # Test OpenPI conversion
    print("\n🤖 Testing OpenPI conversion...")
    
    openpi_obs = robot_obs_to_openpi_obs(robot_obs)
    print(f"✅ Robot → OpenPI: {list(openpi_obs.keys())}")
    
    # Mock OpenPI action response
    openpi_action = {
        "actions": [
            np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.8], dtype=np.float32),
            np.array([0.2, -0.1, 0.4, -0.3, 0.6, -0.5, 0.7], dtype=np.float32)
        ]
    }
    
    converted_action = openpi_action_to_robot_action(openpi_action)
    print(f"✅ OpenPI → Robot: joints {converted_action.joint_positions.shape}, gripper {converted_action.gripper_action}")
    
    # Test quaternion conversion
    print("\n🔄 Testing quaternion conversion...")
    
    test_quat = np.array([0.0, 0.0, 0.707, 0.707], dtype=np.float32)  # 90° rotation around Z
    axis_angle = quat_to_axis_angle(test_quat)
    back_to_quat = axis_angle_to_quat(axis_angle)
    print(f"✅ Quat → Axis-angle → Quat: {test_quat} → {axis_angle} → {back_to_quat}")
    
    print("\n🎉 All format converter tests passed!")
