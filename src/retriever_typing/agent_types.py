#!/usr/bin/env python3
"""
agent_types.py - Standard Types for Observations → Actions Agent Interface

Defines the universal agent interface: Flow[Observations, Actions]
All robot agents implement this standard contract.
"""

import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum

# ======================= OBSERVATION TYPES =======================

@dataclass
class RGBImage:
    """RGB camera image"""
    data: np.ndarray  # Shape: (H, W, 3), dtype: uint8
    timestamp: float
    camera_id: str = "default"
    calibration: Optional[Dict[str, Any]] = None

@dataclass 
class DepthImage:
    """Depth camera image"""
    data: np.ndarray  # Shape: (H, W), dtype: float32, values in meters
    timestamp: float
    camera_id: str = "default"
    calibration: Optional[Dict[str, Any]] = None

@dataclass
class PointCloud:
    """3D point cloud data"""
    points: np.ndarray  # Shape: (N, 3), dtype: float32, XYZ coordinates
    colors: Optional[np.ndarray] = None  # Shape: (N, 3), RGB values
    timestamp: float = 0.0
    frame_id: str = "base_link"

@dataclass
class IMUReading:
    """Inertial measurement unit data"""
    linear_acceleration: np.ndarray  # Shape: (3,), m/s²
    angular_velocity: np.ndarray     # Shape: (3,), rad/s
    orientation: np.ndarray          # Shape: (4,), quaternion (w,x,y,z)
    timestamp: float

@dataclass
class JointStates:
    """Robot joint state information"""
    positions: np.ndarray     # Joint positions (rad or m)
    velocities: np.ndarray    # Joint velocities (rad/s or m/s)  
    efforts: np.ndarray       # Joint torques/forces (Nm or N)
    joint_names: List[str]    # Names of joints
    timestamp: float

@dataclass
class AudioStream:
    """Audio input data"""
    audio_data: np.ndarray    # Raw audio samples
    sample_rate: int          # Samples per second
    channels: int = 1         # Number of audio channels
    timestamp: float = 0.0

@dataclass
class Observations:
    """Complete multi-modal sensor observations"""
    # Visual sensors
    rgb_images: List[RGBImage] = None
    depth_images: List[DepthImage] = None
    
    # Spatial sensors
    lidar_points: Optional[PointCloud] = None
    imu_data: Optional[IMUReading] = None
    
    # Robot state
    joint_states: Optional[JointStates] = None
    
    # Interaction modalities
    audio: Optional[AudioStream] = None
    language_command: Optional[str] = None
    
    # Navigation
    navigation_goal: Optional['NavGoal'] = None
    
    # Metadata
    timestamp: float = 0.0
    robot_id: str = "default"

# ======================= ACTION TYPES =======================

@dataclass
class Twist:
    """Velocity command for mobile base"""
    linear: np.ndarray   # Shape: (3,), linear velocity (m/s) 
    angular: np.ndarray  # Shape: (3,), angular velocity (rad/s)

@dataclass
class ArmCommands:
    """Arm control commands"""
    joint_positions: Optional[np.ndarray] = None  # Target positions
    joint_velocities: Optional[np.ndarray] = None  # Target velocities
    joint_efforts: Optional[np.ndarray] = None     # Target torques/forces
    end_effector_pose: Optional[np.ndarray] = None # Target EE pose (7D: xyz + quat)
    joint_names: List[str] = None

@dataclass
class GripperCommands:
    """Gripper control commands"""
    position: float      # Gripper opening (0=closed, 1=open)
    force: float = 0.5   # Grip force (0-1)
    speed: float = 0.5   # Movement speed (0-1)

@dataclass
class HeadCommands:
    """Head/camera positioning commands"""
    pan: float = 0.0     # Pan angle (rad)
    tilt: float = 0.0    # Tilt angle (rad)
    roll: float = 0.0    # Roll angle (rad)

@dataclass
class NavGoal:
    """Navigation goal specification"""
    target_pose: np.ndarray      # Shape: (7,), xyz + quaternion
    tolerance: float = 0.1       # Position tolerance (m)
    frame_id: str = "map"        # Reference frame

class AgentStatus(Enum):
    NORMAL = "normal"
    CAUTION = "caution"
    ERROR = "error"
    EMERGENCY = "emergency"

@dataclass
class Actions:
    """Complete robot action commands"""
    # Motion commands
    base_velocity: Optional[Twist] = None
    arm_commands: Optional[ArmCommands] = None
    gripper_commands: Optional[GripperCommands] = None
    head_commands: Optional[HeadCommands] = None
    
    # High-level commands
    navigation_goal: Optional[NavGoal] = None
    
    # Communication  
    speech_output: Optional[str] = None
    display_text: Optional[str] = None
    
    # Status and diagnostics
    status: AgentStatus = AgentStatus.NORMAL
    status_message: str = "Operating normally"
    
    # Metadata
    timestamp: float = 0.0
    agent_id: str = "default"

# ======================= HELPER FUNCTIONS =======================

def create_mock_observations(
    include_vision: bool = True,
    include_lidar: bool = False, 
    include_audio: bool = False,
    language_command: Optional[str] = None
) -> Observations:
    """Create mock observations for testing"""
    obs = Observations(timestamp=0.0)
    
    if include_vision:
        # Mock RGB camera
        rgb_data = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        obs.rgb_images = [RGBImage(data=rgb_data, timestamp=0.0)]
        
        # Mock depth camera
        depth_data = np.random.uniform(0.5, 5.0, (480, 640)).astype(np.float32)
        obs.depth_images = [DepthImage(data=depth_data, timestamp=0.0)]
    
    if include_lidar:
        # Mock LiDAR points
        points = np.random.uniform(-10, 10, (1000, 3)).astype(np.float32)
        obs.lidar_points = PointCloud(points=points, timestamp=0.0)
    
    if include_audio:
        # Mock audio stream
        audio_data = np.random.uniform(-1, 1, 16000).astype(np.float32)  # 1 second at 16kHz
        obs.audio = AudioStream(audio_data=audio_data, sample_rate=16000, timestamp=0.0)
    
    if language_command:
        obs.language_command = language_command
    
    # Mock robot state
    obs.joint_states = JointStates(
        positions=np.zeros(7),
        velocities=np.zeros(7), 
        efforts=np.zeros(7),
        joint_names=[f"joint_{i}" for i in range(7)],
        timestamp=0.0
    )
    
    return obs

def create_stop_actions(reason: str = "Safety stop") -> Actions:
    """Create safe stop actions"""
    return Actions(
        base_velocity=Twist(linear=np.zeros(3), angular=np.zeros(3)),
        status=AgentStatus.CAUTION,
        status_message=reason,
        timestamp=0.0
    )