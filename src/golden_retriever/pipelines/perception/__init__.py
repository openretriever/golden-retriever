"""
Perception pipelines combining vision and sensing flows.

This module contains pre-built pipelines for:
- Object detection and tracking
- Scene understanding and mapping
- Visual-inertial odometry
- Multi-modal perception
- Real-time processing pipelines
"""

from .detection import *  # noqa: F401, F403
from .tracking import *  # noqa: F401, F403
from .mapping import *  # noqa: F401, F403
from .multimodal import *  # noqa: F401, F403

__all__ = [
    # Detection pipelines
    "ObjectDetectionPipeline",
    "RealTimeDetectionPipeline",
    "MultiCameraDetectionPipeline",
    # Tracking pipelines
    "ObjectTrackingPipeline",
    "PersonTrackingPipeline",
    "MultiObjectTrackingPipeline",
    # Mapping pipelines
    "SLAMPipeline",
    "SemanticMappingPipeline",
    "VisualOdometryPipeline",
    # Multi-modal pipelines
    "VisionLanguagePipeline",
    "AudioVisualPipeline",
    "RGBDProcessingPipeline",
]