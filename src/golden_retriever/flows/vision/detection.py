"""
Object detection flows for computer vision tasks.

This module provides reusable detection flows for various object detection
models and approaches.
"""

import time
from typing import List, Dict, Any, Optional
import cv2
import numpy as np

from retriever.core.flow import Flow
from retriever.core.types import ExecutionTimer
from retriever.types.core_types import RGBImage, Detection, BoundingBox as BBox
from retriever.core.frp import flow


class DetectionConfig:
    """Configuration for detection operations."""
    def __init__(
        self,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        max_detections: int = 100,
        class_filter: Optional[List[str]] = None
    ):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.max_detections = max_detections
        self.class_filter = class_filter


@flow(rate="10hz")
class ObjectDetectionFlow(Flow[RGBImage, List[Detection]]):
    """Generic object detection flow."""
    
    def __init__(self, config: Optional[DetectionConfig] = None):
        self.config = config or DetectionConfig()
        
    def run_timed(self, image: RGBImage, timer: ExecutionTimer) -> List[Detection]:
        """Run object detection on input image."""
        # This is a base implementation - subclasses should override
        return self._detect_objects(image)
    
    def _detect_objects(self, image: RGBImage) -> List[Detection]:
        """Base detection method to be overridden by specific implementations."""
        # Mock detection for base class
        detections = []
        
        # Simple color-based detection as placeholder
        hsv = cv2.cvtColor(image.data, cv2.COLOR_RGB2HSV)
        
        # Detect red objects
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask = cv2.inRange(hsv, lower_red, upper_red)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                
                detection = Detection(
                    label="red_object",
                    confidence=0.8,
                    bbox=BBox(x=x, y=y, width=w, height=h),
                    timestamp=time.time()
                )
                detections.append(detection)
                
        return detections


@flow(rate="10hz")
class YOLOFlow(ObjectDetectionFlow):
    """YOLO-based object detection flow."""
    
    def __init__(
        self, 
        model_path: Optional[str] = None,
        config: Optional[DetectionConfig] = None
    ):
        super().__init__(config)
        self.model_path = model_path
        self.model = None
        
    def _detect_objects(self, image: RGBImage) -> List[Detection]:
        """YOLO detection implementation."""
        if self.model is None:
            # In practice, load YOLO model here
            # self.model = cv2.dnn.readNet(self.model_path)
            pass
            
        # Mock YOLO detection for now
        detections = []
        
        # Simulate YOLO output
        height, width = image.data.shape[:2]
        
        # Mock detection: person in center of image
        center_x, center_y = width // 2, height // 2
        box_w, box_h = 100, 200
        
        detection = Detection(
            label="person",
            confidence=0.92,
            bbox=BBox(
                x=center_x - box_w // 2,
                y=center_y - box_h // 2,
                width=box_w,
                height=box_h
            ),
            timestamp=time.time()
        )
        detections.append(detection)
        
        return detections


@flow(rate="5hz")  # Slower rate for more complex model
class GroundingDINOFlow(ObjectDetectionFlow):
    """GroundingDINO-based detection with text prompts."""
    
    def __init__(
        self,
        text_prompt: str = "person . car . chair",
        config: Optional[DetectionConfig] = None
    ):
        super().__init__(config)
        self.text_prompt = text_prompt
        self.model = None
        
    def _detect_objects(self, image: RGBImage) -> List[Detection]:
        """GroundingDINO detection with text prompts."""
        if self.model is None:
            # In practice, load GroundingDINO model here
            pass
            
        # Mock GroundingDINO detection
        detections = []
        
        # Simulate text-guided detection
        height, width = image.data.shape[:2]
        
        # Parse text prompt
        classes = [cls.strip() for cls in self.text_prompt.split('.') if cls.strip()]
        
        for i, class_name in enumerate(classes[:3]):  # Limit to 3 for mock
            # Mock bounding box
            x = (i + 1) * width // (len(classes) + 1) - 50
            y = height // 2 - 50
            
            detection = Detection(
                label=class_name,
                confidence=0.85 - i * 0.1,  # Decreasing confidence
                bbox=BBox(x=x, y=y, width=100, height=100),
                timestamp=time.time()
            )
            detections.append(detection)
            
        return detections


class DetectionFilterFlow(Flow[List[Detection], List[Detection]]):
    """Flow for filtering and post-processing detections."""
    
    def __init__(
        self,
        min_confidence: float = 0.5,
        class_whitelist: Optional[List[str]] = None,
        max_detections: int = 50
    ):
        self.min_confidence = min_confidence
        self.class_whitelist = class_whitelist
        self.max_detections = max_detections
        
    def run_timed(self, detections: List[Detection], timer: ExecutionTimer) -> List[Detection]:
        """Filter detections based on criteria."""
        filtered = []
        
        for detection in detections:
            # Confidence filter
            if detection.confidence < self.min_confidence:
                continue
                
            # Class filter
            if self.class_whitelist and detection.label not in self.class_whitelist:
                continue
                
            filtered.append(detection)
            
        # Limit number of detections
        filtered = sorted(filtered, key=lambda d: d.confidence, reverse=True)
        return filtered[:self.max_detections]


class DetectionNMSFlow(Flow[List[Detection], List[Detection]]):
    """Non-Maximum Suppression flow for overlapping detections."""
    
    def __init__(self, iou_threshold: float = 0.5):
        self.iou_threshold = iou_threshold
        
    def run_timed(self, detections: List[Detection], timer: ExecutionTimer) -> List[Detection]:
        """Apply NMS to remove overlapping detections."""
        if len(detections) <= 1:
            return detections
            
        # Convert to format suitable for NMS
        boxes = []
        scores = []
        class_ids = []
        
        for detection in detections:
            bbox = detection.bbox
            boxes.append([bbox.x, bbox.y, bbox.x + bbox.width, bbox.y + bbox.height])
            scores.append(detection.confidence)
            class_ids.append(hash(detection.label) % 1000)  # Simple class ID
            
        boxes = np.array(boxes, dtype=np.float32)
        scores = np.array(scores, dtype=np.float32)
        class_ids = np.array(class_ids, dtype=np.int32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            scores.tolist(),
            score_threshold=0.0,  # Already filtered by confidence
            nms_threshold=self.iou_threshold
        )
        
        if len(indices) == 0:
            return []
            
        # Return filtered detections
        return [detections[i] for i in indices.flatten()]


__all__ = [
    "ObjectDetectionFlow",
    "YOLOFlow",
    "GroundingDINOFlow", 
    "DetectionFilterFlow",
    "DetectionNMSFlow",
    "DetectionConfig",
]