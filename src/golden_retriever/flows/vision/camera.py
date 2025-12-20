"""
Camera flows for image capture and processing.

This module provides reusable camera flows for different types of image acquisition
and basic image processing operations.
"""

import time
from typing import Optional, Tuple
import cv2
import numpy as np

from retriever.core.flow import Flow
from retriever.types.core_types import RGBImage, RGBDImage, ExecutionTimer
from retriever import flow


class CameraInput:
    """Input specification for camera operations."""
    def __init__(
        self, 
        camera_id: int = 0, 
        resolution: Tuple[int, int] = (640, 480),
        fps: int = 30
    ):
        self.camera_id = camera_id
        self.resolution = resolution
        self.fps = fps


@flow(rate="30hz")
class CameraFlow(Flow[None, RGBImage]):
    """Basic RGB camera capture flow."""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        
    def __enter__(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cap:
            self.cap.release()
            
    def run_timed(self, input_data: None, timer: ExecutionTimer) -> RGBImage:
        """Capture RGB image from camera."""
        if not self.cap:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to capture from camera {self.camera_id}")
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return RGBImage(
            data=rgb_frame,
            timestamp=time.time(),
            height=rgb_frame.shape[0],
            width=rgb_frame.shape[1]
        )


@flow(rate="30hz") 
class RGBCameraFlow(CameraFlow):
    """Specialized RGB camera flow with enhanced features."""
    
    def __init__(
        self, 
        camera_id: int = 0, 
        resolution: Tuple[int, int] = (640, 480),
        auto_exposure: bool = True,
        brightness: float = 0.5
    ):
        super().__init__(camera_id, resolution)
        self.auto_exposure = auto_exposure
        self.brightness = brightness
        
    def run_timed(self, input_data: None, timer: ExecutionTimer) -> RGBImage:
        """Capture RGB image with enhanced camera settings."""
        if not self.cap:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
            # Set camera properties
            if not self.auto_exposure:
                self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
            
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to capture from camera {self.camera_id}")
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        return RGBImage(
            data=rgb_frame,
            timestamp=time.time(),
            height=rgb_frame.shape[0],
            width=rgb_frame.shape[1]
        )


@flow(rate="15hz")
class DepthCameraFlow(Flow[None, RGBDImage]):
    """RGBD camera flow for depth-enabled cameras."""
    
    def __init__(self, camera_id: int = 0, resolution: Tuple[int, int] = (640, 480)):
        self.camera_id = camera_id
        self.resolution = resolution
        self.cap: Optional[cv2.VideoCapture] = None
        
    def run_timed(self, input_data: None, timer: ExecutionTimer) -> RGBDImage:
        """Capture RGBD image from depth camera."""
        # This is a mock implementation - in practice would use RealSense, etc.
        if not self.cap:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Failed to capture from camera {self.camera_id}")
            
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Mock depth data (in practice would come from depth sensor)
        depth_frame = np.ones((rgb_frame.shape[0], rgb_frame.shape[1]), dtype=np.float32) * 1000.0
        
        return RGBDImage(
            rgb_data=rgb_frame,
            depth_data=depth_frame,
            timestamp=time.time(),
            height=rgb_frame.shape[0],
            width=rgb_frame.shape[1]
        )


class ImageProcessingFlow(Flow[RGBImage, RGBImage]):
    """Basic image processing flow for enhancement and filtering."""
    
    def __init__(
        self,
        blur_kernel: Tuple[int, int] = (5, 5),
        sharpen: bool = False,
        denoise: bool = False
    ):
        self.blur_kernel = blur_kernel
        self.sharpen = sharpen
        self.denoise = denoise
        
    def run_timed(self, image: RGBImage, timer: ExecutionTimer) -> RGBImage:
        """Process input image with specified enhancements."""
        processed = image.data.copy()
        
        # Apply Gaussian blur
        if self.blur_kernel[0] > 1 and self.blur_kernel[1] > 1:
            processed = cv2.GaussianBlur(processed, self.blur_kernel, 0)
            
        # Apply sharpening
        if self.sharpen:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            processed = cv2.filter2D(processed, -1, kernel)
            
        # Apply denoising
        if self.denoise:
            processed = cv2.fastNlMeansDenoisingColored(processed)
            
        return RGBImage(
            data=processed,
            timestamp=time.time(),
            height=processed.shape[0],
            width=processed.shape[1]
        )


__all__ = [
    "CameraFlow",
    "RGBCameraFlow", 
    "DepthCameraFlow",
    "ImageProcessingFlow",
    "CameraInput",
]