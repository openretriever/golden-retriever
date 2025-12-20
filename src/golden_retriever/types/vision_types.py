"""
Computer Vision Data Types

Types specifically for computer vision applications including 
segmentation, detection enhancement, and multi-modal data.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from .core_types import Detection, RGBImage
from .registry import register_type


# ============================================================================
# Enhanced Image Types
# ============================================================================

@register_type(description="RGB-D image pair")
@dataclass
class RGBDImage:
    """A single RGB-D image pair."""
    rgb: np.ndarray  # HxWx3, uint8
    depth: np.ndarray  # HxW, float32 meters
    timestamp: Optional[float] = None
    camera_id: str = "default"
    
    @property
    def shape(self) -> tuple:
        """Get RGB image shape."""
        return self.rgb.shape


# ============================================================================
# Segmentation Types
# ============================================================================

@register_type(description="Segmentation mask")
@dataclass
class SegmentationMask:
    """Segmentation mask result."""
    mask: np.ndarray  # Binary or labeled mask
    label: str
    confidence: float
    class_id: Optional[int] = None
    
    @property
    def area(self) -> int:
        """Get mask area (number of pixels)."""
        return int(np.sum(self.mask > 0))


# ============================================================================
# Environment and Observation Types
# ============================================================================

@register_type(description="Environment observation")
@dataclass
class EnvironmentObservation:
    """Observation from a robotic environment."""
    color: List[np.ndarray]  # List of RGB images from different cameras
    depth: List[np.ndarray]  # List of depth images from different cameras
    metadata: Dict[str, Any]  # Additional environment information
    timestamp: Optional[float] = None
    
    @property
    def num_cameras(self) -> int:
        """Get number of cameras."""
        return len(self.color)


# ============================================================================
# VLM and Language Types
# ============================================================================

@register_type(description="Vision-Language Model response")
@dataclass 
class VLMResponse:
    """Response from a Vision-Language Model."""
    content: str
    metadata: Dict[str, Any]
    confidence: Optional[float] = None
    tokens_used: Optional[int] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@register_type(description="Natural language command")
@dataclass
class NLCommand:
    """A natural language command."""
    text: str
    language: str = "en"
    confidence: Optional[float] = None
    parsed_intent: Optional[Dict[str, Any]] = None


# ============================================================================
# Web Interface Types
# ============================================================================

@register_type(description="Web application response")
@dataclass
class WebResponse:
    """Response from web application pipeline."""
    detections: List[Detection]
    instruction: str
    timestamp: float
    success: bool = True
    error_message: Optional[str] = None
    processing_time: Optional[float] = None


# ============================================================================
# Ray Actor Types
# ============================================================================

@register_type(description="Ray actor handle wrapper")
@dataclass
class ActorHandle:
    """Type-safe wrapper for Ray actor handles."""
    handle: Any  # The actual Ray actor handle
    name: str
    actor_type: str = "generic"