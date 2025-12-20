"""
Vision visualization components with Flow registry integration.

Components automatically register themselves for easy discovery:
- camera = get_flow("camera") 
- detector = get_flow("color_detector")
- visualizer = get_flow("opencv_visualizer")
"""

import cv2
import numpy as np
import time
import threading
from typing import List, Optional, TYPE_CHECKING
from collections import deque

if TYPE_CHECKING:
    # Import types for static type checking only
    from retriever.types.core_types import RGBImage, Detection, BoundingBox
else:
    # Runtime imports via registry
    from retriever import get_type
    RGBImage = get_type('RGBImage')
    Detection = get_type('Detection')
    BoundingBox = get_type('BoundingBox')

from retriever import Flow, register_flow


@register_flow("camera", category="vision", description="Camera with test pattern fallback")
class CameraFlow(Flow[None, RGBImage]):
    """Reusable camera capture with test pattern fallback."""
    
    def __init__(self, width: int = 640, height: int = 480):
        super().__init__()
        self.cap = None
        self.frame_count = 0
        self.width = width
        self.height = height
        
    def run(self, _: None) -> RGBImage:
        """Capture camera frame and return as RGBImage."""
        frame = self.get_frame()
        return RGBImage(data=frame)
        
    def get_frame(self) -> np.ndarray:
        """Get camera frame or test pattern."""
        if self.cap is None:
            self._init_camera()
        
        ret, frame = self.cap.read() if self.cap else (False, None)
        if not ret:
            frame = self._generate_test_pattern()
        
        return frame
    
    def _init_camera(self):
        """Initialize camera with fallback indices."""
        for camera_idx in [0, 1, 2]:
            self.cap = cv2.VideoCapture(camera_idx)
            if self.cap.isOpened():
                ret, test_frame = self.cap.read()
                if ret and test_frame is not None:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    return
                self.cap.release()
                self.cap = None
    
    def _generate_test_pattern(self) -> np.ndarray:
        """Generate test pattern with colored objects."""
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        frame[120:240, 200:320] = [255, 50, 50]   # Red object
        frame[120:240, 400:520] = [50, 255, 50]   # Green object
        cv2.putText(frame, f"TEST PATTERN - Frame {self.frame_count}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        self.frame_count += 1
        return frame
    
    def cleanup(self):
        """Release camera resources."""
        if self.cap is not None:
            self.cap.release()


@register_flow("color_detector", category="vision",
               description="HSV color-based object detector", 
               tags=["detection", "computer_vision"])
class ColorDetector(Flow[RGBImage, List[Detection]]):
    """HSV color-based object detector."""
    
    def __init__(self):
        super().__init__()
        self.color_ranges = {
            "red_object": ([0, 50, 50], [10, 255, 255]),
            "green_object": ([50, 50, 50], [70, 255, 255]), 
            "blue_object": ([100, 50, 50], [130, 255, 255]),
        }
    
    def run(self, image: RGBImage) -> List[Detection]:
        """Detect colored objects in RGB image."""
        return self.detect(image.data)
    
    def detect(self, image: np.ndarray) -> List[Detection]:
        """Detect colored objects in RGB image."""
        detections = []
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        for label, (lower, upper) in self.color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 500:
                    x, y, w, h = cv2.boundingRect(contour)
                    confidence = min(0.9, area / 10000)
                    
                    detection = Detection(
                        label=label,
                        confidence=confidence,
                        bbox=BoundingBox(x=x, y=y, width=w, height=h)
                    )
                    detections.append(detection)
        
        return detections


@register_flow("opencv_visualizer", category="vision", description="OpenCV-based visualizer")
class OpenCVVisualizer:
    """Simple OpenCV visualization with bounding boxes."""
    
    def __init__(self, window_name: str = "Dora Visualization"):
        self.window_name = window_name
        self.frame_count = 0
        
    def show(self, image: RGBImage, detections: List[Detection], info: str = ""):
        """Display image with bounding boxes."""
        try:
            self.frame_count += 1
            display_frame = cv2.cvtColor(image.data, cv2.COLOR_RGB2BGR)
            
            # Draw bounding boxes
            for det in detections:
                bbox = det.bbox
                x, y, w, h = int(bbox.x), int(bbox.y), int(bbox.width), int(bbox.height)
                
                color = (0, 255, 0)  # Green default
                if 'red' in det.label:
                    color = (0, 0, 255)
                elif 'blue' in det.label:
                    color = (255, 0, 0)
                
                cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display_frame, f"{det.label}: {det.confidence:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add info text
            if info:
                cv2.putText(display_frame, info, (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.putText(display_frame, f"Frame {self.frame_count} | Detections: {len(detections)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press 'q' to quit", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            cv2.imshow(self.window_name, display_frame)
            return cv2.waitKey(1) & 0xFF
            
        except Exception:
            # Silently handle OpenCV exceptions
            return -1
    
    def cleanup(self):
        """Clean up OpenCV windows."""
        cv2.destroyAllWindows()