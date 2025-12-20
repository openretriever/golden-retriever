"""
Object detection pipelines combining camera and detection flows.

This module provides pre-built pipelines for real-time object detection
using different camera sources and detection models.
"""

from typing import List, Optional, Tuple
import time

from retriever.core.flow import Flow
from retriever.core.types import Pipeline, ExecutionTimer
from retriever.types.core_types import RGBImage, Detection
from retriever.core.frp import flow
from retriever.flows.vision.camera import CameraFlow, RGBCameraFlow
from retriever.flows.vision.detection import (
    ObjectDetectionFlow, YOLOFlow, GroundingDINOFlow,
    DetectionFilterFlow, DetectionNMSFlow, DetectionConfig
)


@flow(rate="10hz")
class ObjectDetectionPipeline(Flow[None, List[Detection]]):
    """Complete object detection pipeline from camera to filtered detections."""
    
    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        detection_model: str = "yolo",
        confidence_threshold: float = 0.5,
        apply_nms: bool = True,
        nms_threshold: float = 0.5
    ):
        self.camera_id = camera_id
        self.resolution = resolution
        self.detection_model = detection_model
        self.confidence_threshold = confidence_threshold
        self.apply_nms = apply_nms
        self.nms_threshold = nms_threshold
        
        # Initialize components
        self._setup_pipeline()
    
    def _setup_pipeline(self):
        """Initialize pipeline components."""
        # Camera flow
        self.camera_flow = RGBCameraFlow(
            camera_id=self.camera_id,
            resolution=self.resolution
        )
        
        # Detection flow
        config = DetectionConfig(confidence_threshold=self.confidence_threshold)
        if self.detection_model.lower() == "yolo":
            self.detection_flow = YOLOFlow(config=config)
        elif self.detection_model.lower() == "grounding_dino":
            self.detection_flow = GroundingDINOFlow(config=config)
        else:
            self.detection_flow = ObjectDetectionFlow(config=config)
        
        # Post-processing flows
        self.filter_flow = DetectionFilterFlow(min_confidence=self.confidence_threshold)
        if self.apply_nms:
            self.nms_flow = DetectionNMSFlow(iou_threshold=self.nms_threshold)
    
    def run_timed(self, input_data: None, timer: ExecutionTimer) -> List[Detection]:
        """Execute complete detection pipeline."""
        # Capture image
        image = self.camera_flow.run_timed(input_data, timer)
        
        # Run detection
        detections = self.detection_flow.run_timed(image, timer)
        
        # Apply filtering
        filtered_detections = self.filter_flow.run_timed(detections, timer)
        
        # Apply NMS if enabled
        if self.apply_nms:
            final_detections = self.nms_flow.run_timed(filtered_detections, timer)
        else:
            final_detections = filtered_detections
        
        return final_detections


@flow(rate="30hz")
class RealTimeDetectionPipeline(Flow[None, Tuple[RGBImage, List[Detection]]]):
    """Real-time detection pipeline returning both image and detections."""
    
    def __init__(
        self,
        camera_id: int = 0,
        resolution: Tuple[int, int] = (640, 480),
        detection_rate_divisor: int = 3  # Detect every N frames
    ):
        self.camera_id = camera_id
        self.resolution = resolution
        self.detection_rate_divisor = detection_rate_divisor
        self.frame_count = 0
        
        # Initialize components
        self.camera_flow = RGBCameraFlow(
            camera_id=camera_id,
            resolution=resolution
        )
        self.detection_flow = YOLOFlow()
        
        # Cache last detections for high-frequency output
        self.last_detections: List[Detection] = []
    
    def run_timed(self, input_data: None, timer: ExecutionTimer) -> Tuple[RGBImage, List[Detection]]:
        """Execute real-time detection with smart frame skipping."""
        # Always capture image
        image = self.camera_flow.run_timed(input_data, timer)
        
        # Run detection only periodically
        if self.frame_count % self.detection_rate_divisor == 0:
            self.last_detections = self.detection_flow.run_timed(image, timer)
        
        self.frame_count += 1
        
        return image, self.last_detections


class MultiCameraDetectionPipeline(Flow[None, List[Tuple[int, List[Detection]]]]):
    """Multi-camera detection pipeline for comprehensive coverage."""
    
    def __init__(
        self,
        camera_ids: List[int] = [0, 1],
        resolution: Tuple[int, int] = (640, 480),
        detection_model: str = "yolo"
    ):
        self.camera_ids = camera_ids
        self.resolution = resolution
        self.detection_model = detection_model
        
        # Initialize per-camera pipelines
        self.pipelines = {}
        for camera_id in camera_ids:
            pipeline = ObjectDetectionPipeline(
                camera_id=camera_id,
                resolution=resolution,
                detection_model=detection_model
            )
            self.pipelines[camera_id] = pipeline
    
    def run_timed(self, input_data: None, timer: ExecutionTimer) -> List[Tuple[int, List[Detection]]]:
        """Run detection on all cameras."""
        results = []
        
        for camera_id, pipeline in self.pipelines.items():
            try:
                detections = pipeline.run_timed(input_data, timer)
                results.append((camera_id, detections))
            except Exception as e:
                print(f"Error processing camera {camera_id}: {e}")
                results.append((camera_id, []))
        
        return results


# Compositional pipeline using the >> operator from core framework
def create_detection_pipeline(
    camera_id: int = 0,
    detection_model: str = "yolo",
    confidence_threshold: float = 0.5
) -> Pipeline:
    """Create a detection pipeline using compositional operators."""
    
    # Setup flows
    camera = RGBCameraFlow(camera_id=camera_id)
    
    if detection_model.lower() == "yolo":
        detector = YOLOFlow()
    else:
        detector = ObjectDetectionFlow()
    
    filter_flow = DetectionFilterFlow(min_confidence=confidence_threshold)
    nms_flow = DetectionNMSFlow()
    
    # Compose pipeline
    pipeline = camera >> detector >> filter_flow >> nms_flow
    
    return pipeline


__all__ = [
    "ObjectDetectionPipeline",
    "RealTimeDetectionPipeline", 
    "MultiCameraDetectionPipeline",
    "create_detection_pipeline",
]