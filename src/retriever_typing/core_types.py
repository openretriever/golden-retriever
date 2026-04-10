"""
Core Types for Retriever Framework

These are the fundamental types provided by retriever-core package.
External packages can register additional types using the registry system.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Any
from .registry import register_type


# ============================================================================
# Image and Sensor Data Types
# ============================================================================

@register_type(description="RGB image data with metadata")
@dataclass 
class RGBImage:
    """RGB image data."""
    data: np.ndarray  # Shape: (H, W, 3), dtype: uint8
    timestamp: Optional[float] = None
    camera_id: str = "default"
    
    @property
    def shape(self) -> Tuple[int, int, int]:
        """Get image shape as (height, width, channels)."""
        return self.data.shape
    
    @property
    def height(self) -> int:
        return self.data.shape[0]
    
    @property
    def width(self) -> int:
        return self.data.shape[1]
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        import pyarrow as pa
        return {
            "data": pa.array(self.data.flatten()),
            "shape": self.data.shape,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id
        }
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'RGBImage':
        """Convert from Arrow format."""
        import numpy as np
        data = np.array(arrow_data["data"]).reshape(arrow_data["shape"])
        return cls(
            data=data,
            timestamp=arrow_data.get("timestamp"),
            camera_id=arrow_data.get("camera_id", "default")
        )


@register_type(description="Depth image data with metadata") 
@dataclass
class DepthImage:
    """Depth image data."""
    data: np.ndarray  # Shape: (H, W), dtype: float32, values in meters
    timestamp: Optional[float] = None
    camera_id: str = "default"
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.data.shape
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        import pyarrow as pa
        return {
            "data": pa.array(self.data.flatten()),
            "shape": self.data.shape,
            "timestamp": self.timestamp,
            "camera_id": self.camera_id
        }
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'DepthImage':
        """Convert from Arrow format."""
        import numpy as np
        data = np.array(arrow_data["data"]).reshape(arrow_data["shape"])
        return cls(
            data=data,
            timestamp=arrow_data.get("timestamp"),
            camera_id=arrow_data.get("camera_id", "default")
        )


@register_type(description="3D point cloud data")
@dataclass  
class PointCloud:
    """3D point cloud data."""
    points: np.ndarray  # Shape: (N, 3), dtype: float32, XYZ coordinates
    colors: Optional[np.ndarray] = None  # Shape: (N, 3), dtype: uint8, RGB values
    timestamp: Optional[float] = None
    frame_id: str = "world"
    
    @property
    def num_points(self) -> int:
        return self.points.shape[0]
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        import pyarrow as pa
        result = {
            "points": pa.array(self.points.flatten()),
            "points_shape": self.points.shape,
            "timestamp": self.timestamp,
            "frame_id": self.frame_id
        }
        if self.colors is not None:
            result["colors"] = pa.array(self.colors.flatten())
            result["colors_shape"] = self.colors.shape
        return result
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'PointCloud':
        """Convert from Arrow format."""
        import numpy as np
        points = np.array(arrow_data["points"]).reshape(arrow_data["points_shape"])
        
        colors = None
        if "colors" in arrow_data:
            colors = np.array(arrow_data["colors"]).reshape(arrow_data["colors_shape"])
        
        return cls(
            points=points,
            colors=colors,
            timestamp=arrow_data.get("timestamp"),
            frame_id=arrow_data.get("frame_id", "world")
        )


# ============================================================================
# Detection and Vision Types  
# ============================================================================

@register_type(description="2D bounding box")
@dataclass
class BoundingBox:
    """2D bounding box."""
    x: float  # Top-left x coordinate
    y: float  # Top-left y coordinate  
    width: float
    height: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get center coordinates."""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> float:
        """Get bounding box area."""
        return self.width * self.height
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        return {
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height
        }
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'BoundingBox':
        """Convert from Arrow format."""
        return cls(
            x=arrow_data["x"],
            y=arrow_data["y"],
            width=arrow_data["width"],
            height=arrow_data["height"]
        )


@register_type(description="Object detection result")
@dataclass
class Detection:
    """Object detection result."""
    label: str
    confidence: float
    bbox: BoundingBox
    mask: Optional[np.ndarray] = None  # Binary mask for segmentation
    features: Optional[np.ndarray] = None  # Feature vector
    
    def __post_init__(self):
        """Validate detection data."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence must be in [0, 1], got {self.confidence}")
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        import pyarrow as pa
        result = {
            "label": self.label,
            "confidence": self.confidence,
            "bbox": self.bbox.to_arrow()
        }
        if self.mask is not None:
            result["mask"] = pa.array(self.mask.flatten())
            result["mask_shape"] = self.mask.shape
        if self.features is not None:
            result["features"] = pa.array(self.features.flatten())
            result["features_shape"] = self.features.shape
        return result
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'Detection':
        """Convert from Arrow format."""
        import numpy as np
        bbox = BoundingBox.from_arrow(arrow_data["bbox"])
        
        mask = None
        if "mask" in arrow_data:
            mask = np.array(arrow_data["mask"]).reshape(arrow_data["mask_shape"])
        
        features = None
        if "features" in arrow_data:
            features = np.array(arrow_data["features"]).reshape(arrow_data["features_shape"])
        
        return cls(
            label=arrow_data["label"],
            confidence=arrow_data["confidence"],
            bbox=bbox,
            mask=mask,
            features=features
        )


# ============================================================================
# Spatial and Geometric Types
# ============================================================================

@register_type(description="3D pose (position + orientation)")
@dataclass
class Pose3:
    """3D pose representation."""
    position: np.ndarray  # Shape: (3,), XYZ coordinates
    orientation: np.ndarray  # Shape: (4,), quaternion (w, x, y, z)
    frame_id: str = "world"
    
    def __post_init__(self):
        """Validate pose data."""
        if self.position.shape != (3,):
            raise ValueError(f"Position must be shape (3,), got {self.position.shape}")
        if self.orientation.shape != (4,):
            raise ValueError(f"Orientation must be shape (4,), got {self.orientation.shape}")
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        import pyarrow as pa
        return {
            "position": pa.array(self.position),
            "orientation": pa.array(self.orientation),
            "frame_id": self.frame_id
        }
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'Pose3':
        """Convert from Arrow format."""
        import numpy as np
        return cls(
            position=np.array(arrow_data["position"]),
            orientation=np.array(arrow_data["orientation"]),
            frame_id=arrow_data.get("frame_id", "world")
        )


@register_type(description="3D transformation matrix")
@dataclass
class Transform3:
    """3D transformation matrix."""
    matrix: np.ndarray  # Shape: (4, 4), homogeneous transformation matrix
    from_frame: str = "world"
    to_frame: str = "base"
    
    def __post_init__(self):
        """Validate transformation matrix."""
        if self.matrix.shape != (4, 4):
            raise ValueError(f"Transform matrix must be shape (4, 4), got {self.matrix.shape}")
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        import pyarrow as pa
        return {
            "matrix": pa.array(self.matrix.flatten()),
            "from_frame": self.from_frame,
            "to_frame": self.to_frame
        }
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'Transform3':
        """Convert from Arrow format."""
        import numpy as np
        matrix = np.array(arrow_data["matrix"]).reshape((4, 4))
        return cls(
            matrix=matrix,
            from_frame=arrow_data.get("from_frame", "world"),
            to_frame=arrow_data.get("to_frame", "base")
        )


# ============================================================================
# Robot Action and Control Types
# ============================================================================

@register_type(description="Robot action command")
@dataclass
class Action:
    """Generic robot action."""
    type: str  # e.g., "move", "grasp", "release"
    parameters: dict  # Action-specific parameters
    timestamp: Optional[float] = None
    priority: int = 0  # Higher priority = more urgent
    
    def __post_init__(self):
        """Ensure parameters is a dict."""
        if not isinstance(self.parameters, dict):
            raise ValueError("Action parameters must be a dict")
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        import json
        return {
            "type": self.type,
            "parameters": json.dumps(self.parameters),
            "timestamp": self.timestamp,
            "priority": self.priority
        }
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'Action':
        """Convert from Arrow format."""
        import json
        return cls(
            type=arrow_data["type"],
            parameters=json.loads(arrow_data["parameters"]),
            timestamp=arrow_data.get("timestamp"),
            priority=arrow_data.get("priority", 0)
        )


@register_type(description="Robot command with execution info")
@dataclass
class Command:
    """Robot command with execution metadata."""
    action: Action
    robot_id: str = "default"
    expected_duration: Optional[float] = None  # Expected execution time in seconds
    timeout: Optional[float] = None  # Maximum execution time
    
    @property
    def action_type(self) -> str:
        """Get the action type."""
        return self.action.type
    
    def to_arrow(self) -> dict:
        """Convert to Arrow-compatible format."""
        return {
            "action": self.action.to_arrow(),
            "robot_id": self.robot_id,
            "expected_duration": self.expected_duration,
            "timeout": self.timeout
        }
    
    @classmethod
    def from_arrow(cls, arrow_data: dict) -> 'Command':
        """Convert from Arrow format."""
        return cls(
            action=Action.from_arrow(arrow_data["action"]),
            robot_id=arrow_data.get("robot_id", "default"),
            expected_duration=arrow_data.get("expected_duration"),
            timeout=arrow_data.get("timeout")
        )


@register_type(description="Execution status information")
@dataclass
class Status:
    """Execution status."""
    state: str  # "pending", "running", "completed", "failed", "cancelled"
    message: str = ""
    progress: Optional[float] = None  # Progress percentage (0-100)
    timestamp: Optional[float] = None
    error_code: Optional[int] = None
    
    @property
    def is_complete(self) -> bool:
        """Check if execution is complete (success or failure)."""
        return self.state in ["completed", "failed", "cancelled"]
    
    @property
    def is_success(self) -> bool:
        """Check if execution was successful."""
        return self.state == "completed"


# ============================================================================
# Temporal and Execution Types
# ============================================================================

@register_type(description="High-precision timestamp")
@dataclass
class Timestamp:
    """High-precision timestamp."""
    seconds: int
    nanoseconds: int
    
    @classmethod
    def now(cls) -> 'Timestamp':
        """Create timestamp for current time."""
        import time
        t = time.time()
        return cls(seconds=int(t), nanoseconds=int((t % 1) * 1e9))
    
    def to_float(self) -> float:
        """Convert to floating-point seconds."""
        return self.seconds + self.nanoseconds / 1e9


@register_type(description="Execution timing information")
@dataclass
class ExecutionTimer:
    """Execution timing information."""
    start_time: Timestamp
    expected_period: Optional[float] = None  # Expected period in seconds
    actual_period: Optional[float] = None    # Actual period in seconds
    iteration: int = 0
    
    @property
    def is_timing_violation(self) -> bool:
        """Check if there's a timing violation."""
        if self.expected_period is None or self.actual_period is None:
            return False
        return self.actual_period > self.expected_period * 1.5  # 50% tolerance