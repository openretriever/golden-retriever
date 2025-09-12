#!/usr/bin/env python3
"""
Robotics Data Types

Standard observation and action types for robotics applications.
These types work across different environments (Libero, DROID, etc.) and policies (OpenPI, RT-1, etc.).
"""

import dataclasses
from typing import Dict, Any, Optional
import numpy as np


@dataclasses.dataclass
class RobotObservation:
    """
    Standard robot observation format.
    
    This unified format allows different environments to work with the same controllers.
    """
    
    # Visual observations - dict of camera name to RGB image
    images: Dict[str, np.ndarray]  # e.g., {"agentview": (H,W,3), "wrist": (H,W,3)}
    
    # Robot state - joint positions, gripper state, etc.
    robot_state: np.ndarray  # Shape: (state_dim,) - typically joint positions + gripper
    
    # Task description in natural language
    task_info: str
    
    # Additional environment-specific data
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate observation format"""
        # Ensure images are uint8 RGB
        for cam_name, img in self.images.items():
            if not isinstance(img, np.ndarray):
                raise ValueError(f"Image {cam_name} must be numpy array")
            if img.dtype != np.uint8:
                raise ValueError(f"Image {cam_name} must be uint8")
            if len(img.shape) != 3 or img.shape[2] != 3:
                raise ValueError(f"Image {cam_name} must be (H,W,3) RGB format")
        
        # Ensure robot state is float32
        if not isinstance(self.robot_state, np.ndarray):
            self.robot_state = np.array(self.robot_state, dtype=np.float32)
        elif self.robot_state.dtype != np.float32:
            self.robot_state = self.robot_state.astype(np.float32)


@dataclasses.dataclass  
class RobotAction:
    """
    Standard robot action format.
    
    This unified format allows different controllers to work with the same environments.
    """
    
    # Joint target positions or velocities
    joint_positions: np.ndarray  # Shape: (7,) for 7-DOF arm
    
    # Gripper action: -1 (close) to +1 (open)
    gripper_action: float
    
    # Additional action information
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate action format"""
        # Ensure joint positions are float32
        if not isinstance(self.joint_positions, np.ndarray):
            self.joint_positions = np.array(self.joint_positions, dtype=np.float32)
        elif self.joint_positions.dtype != np.float32:
            self.joint_positions = self.joint_positions.astype(np.float32)
        
        # Ensure gripper is float
        self.gripper_action = float(self.gripper_action)
        
        # Clamp gripper to valid range
        self.gripper_action = max(-1.0, min(1.0, self.gripper_action))
    
    def to_libero_format(self) -> list:
        """Convert to Libero action format: [x, y, z, rx, ry, rz, gripper]"""
        # For now, assume joint_positions contains [x, y, z, rx, ry, rz]
        if len(self.joint_positions) >= 6:
            action = self.joint_positions[:6].tolist() + [self.gripper_action]
        else:
            # Fallback: pad with zeros if needed
            action = list(self.joint_positions) + [0.0] * (6 - len(self.joint_positions)) + [self.gripper_action]
        
        return action
    
    def to_openpi_format(self) -> Dict[str, np.ndarray]:
        """Convert to OpenPI action format"""
        return {
            "robot0_joint_pos": self.joint_positions,
            "robot0_gripper": np.array([self.gripper_action], dtype=np.float32)
        }


# Note: Libero conversion functions moved to format_converters.py
# Import them here for backward compatibility
def libero_obs_to_robot_obs(libero_obs: Dict[str, Any], task_info: str) -> RobotObservation:
    """Convert Libero observation to standard RobotObservation format"""
    from ..converters.format_converters import libero_obs_to_robot_obs as _libero_obs_to_robot_obs
    return _libero_obs_to_robot_obs(libero_obs, task_info)


def robot_action_to_libero_action(robot_action: RobotAction) -> list:
    """Convert RobotAction to Libero action format"""
    return robot_action.to_libero_format()


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle representation"""
    from ..converters.format_converters import quat_to_axis_angle as _quat_to_axis_angle
    return _quat_to_axis_angle(quat)


if __name__ == "__main__":
    # Test the data types
    print("🧪 Testing robotics data types...")
    
    # Test RobotObservation
    test_obs = RobotObservation(
        images={
            "agentview": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8),
            "wrist": np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        },
        robot_state=np.array([0.1, -0.5, 1.2, -0.8, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        task_info="pick up the red cup"
    )
    print(f"✅ RobotObservation: {len(test_obs.images)} cameras, state shape {test_obs.robot_state.shape}")
    
    # Test RobotAction
    test_action = RobotAction(
        joint_positions=np.array([0.1, -0.2, 0.3, -0.4, 0.5, -0.6], dtype=np.float32),
        gripper_action=0.5
    )
    print(f"✅ RobotAction: {test_action.joint_positions.shape} joints, gripper {test_action.gripper_action}")
    
    # Test format conversions
    libero_action = test_action.to_libero_format()
    openpi_action = test_action.to_openpi_format()
    print(f"✅ Libero format: {len(libero_action)} elements")
    print(f"✅ OpenPI format: {list(openpi_action.keys())}")
    
    print("🎉 All tests passed!")
