# Retriever Pipelines

This directory contains pre-built pipeline compositions that combine multiple flows to accomplish complex robotics tasks. Pipelines provide higher-level abstractions for multi-step operations commonly used in robotics applications.

## Organization

### Perception Pipelines (`perception/`)
Vision and sensing pipeline compositions:
- **detection.py**: Object detection and recognition pipelines
- **tracking.py**: Object tracking and temporal consistency
- **mapping.py**: SLAM and spatial understanding pipelines
- **multimodal.py**: Multi-sensor and cross-modal pipelines

### Manipulation Pipelines (`manipulation/`)
Object manipulation and grasping compositions:
- **grasping.py**: Pick and grasp operation pipelines
- **placement.py**: Object placement and positioning
- **bimanual.py**: Dual-arm coordination pipelines
- **assembly.py**: Construction and assembly tasks

### Navigation Pipelines (`navigation/`)
Movement and path planning compositions:
- **pathfinding.py**: Global and local path planning
- **avoidance.py**: Obstacle detection and avoidance
- **exploration.py**: Environment exploration strategies
- **semantic.py**: Language-guided navigation

### Interaction Pipelines (`interaction/`)
Human-robot interaction compositions:
- **language.py**: Natural language processing
- **gesture.py**: Gesture recognition and response
- **social.py**: Social robotics behaviors
- **collaboration.py**: Human-robot collaboration

## Usage Examples

### Complete Detection Pipeline

```python
from retriever.pipelines.perception.detection import ObjectDetectionPipeline

# Create complete detection pipeline
pipeline = ObjectDetectionPipeline(
    camera_id=0,
    detection_model="yolo",
    confidence_threshold=0.7,
    apply_nms=True
)

# Execute pipeline
detections = pipeline.run(None)
```

### Real-time Processing

```python
from retriever.pipelines.perception.detection import RealTimeDetectionPipeline

# High-frequency pipeline with smart frame skipping
pipeline = RealTimeDetectionPipeline(
    camera_id=0,
    detection_rate_divisor=3  # Detect every 3rd frame
)

# Returns both image and detections
image, detections = pipeline.run(None)
```

### Compositional Pipeline Creation

```python
from retriever.pipelines.perception.detection import create_detection_pipeline

# Create using compositional >> operator
pipeline = create_detection_pipeline(
    camera_id=0,
    detection_model="yolo",
    confidence_threshold=0.6
)

# Execute with Pipeline framework
result = pipeline.execute(None)
```

### Multi-Camera Setup

```python
from retriever.pipelines.perception.detection import MultiCameraDetectionPipeline

# Process multiple cameras simultaneously
pipeline = MultiCameraDetectionPipeline(
    camera_ids=[0, 1, 2],
    resolution=(640, 480),
    detection_model="yolo"
)

# Returns list of (camera_id, detections) tuples
results = pipeline.run(None)
```

## Pipeline Development Guidelines

### 1. Flow Composition
Pipelines should compose existing flows rather than implementing logic directly:

```python
class MyPipeline(Flow[Input, Output]):
    def __init__(self):
        self.flow1 = SomeFlow()
        self.flow2 = AnotherFlow()
        self.flow3 = FinalFlow()
    
    def run_timed(self, input_data: Input, timer: ExecutionTimer) -> Output:
        intermediate1 = self.flow1.run_timed(input_data, timer)
        intermediate2 = self.flow2.run_timed(intermediate1, timer)
        return self.flow3.run_timed(intermediate2, timer)
```

### 2. Configuration Support
Provide flexible configuration for different use cases:

```python
class ConfigurablePipeline(Flow[Input, Output]):
    def __init__(
        self,
        model_type: str = "default",
        confidence_threshold: float = 0.5,
        enable_tracking: bool = True,
        **kwargs
    ):
        # Configure based on parameters
        if model_type == "yolo":
            self.detector = YOLOFlow(**kwargs)
        else:
            self.detector = ObjectDetectionFlow(**kwargs)
        
        if enable_tracking:
            self.tracker = TrackingFlow()
```

### 3. Error Recovery
Implement robust error handling and recovery:

```python
def run_timed(self, input_data: Input, timer: ExecutionTimer) -> Output:
    try:
        # Primary processing path
        return self._process_normal(input_data, timer)
    except CameraError:
        # Fallback to cached data
        return self._process_fallback(input_data, timer)
    except ModelError as e:
        # Graceful degradation
        logger.warning(f"Model failed, using backup: {e}")
        return self._process_backup(input_data, timer)
```

### 4. Performance Optimization
Consider performance implications of pipeline design:

```python
class OptimizedPipeline(Flow[Input, Output]):
    def __init__(self):
        # Cache expensive computations
        self._feature_cache = {}
        self._last_computation_time = 0
        
        # Parallel processing where possible
        self.parallel_flows = [Flow1(), Flow2(), Flow3()]
    
    def run_timed(self, input_data: Input, timer: ExecutionTimer) -> Output:
        # Skip expensive computation if recent
        if time.time() - self._last_computation_time < 0.1:
            return self._cached_result
        
        # Process flows in parallel
        results = self._execute_parallel(input_data, timer)
        return self._combine_results(results)
```

### 5. Resource Management
Handle resource lifecycle properly:

```python
class ResourceManagedPipeline(Flow[Input, Output]):
    def __enter__(self):
        self.camera = Camera()
        self.camera.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.camera.stop()
        self.camera.release()
```

## FRP Integration

Pipelines work seamlessly with the FRP (Functional Reactive Programming) system:

```python
@flow(rate="10hz", trigger="camera_frame")
class ReactivePipeline(Flow[Input, Output]):
    def run_timed(self, input_data: Input, timer: ExecutionTimer) -> Output:
        # Reactive processing
        return result

# Dora execution backend
from retriever.core.execution import DoraExecutor

executor = DoraExecutor()
pipeline = ReactivePipeline()
executor.run(pipeline, input_stream=camera_stream)
```

## Testing Pipelines

Test pipelines with mock flows and known inputs:

```python
import pytest
from unittest.mock import Mock
from retriever.pipelines.perception.detection import ObjectDetectionPipeline

def test_detection_pipeline():
    # Mock camera and detector
    mock_camera = Mock()
    mock_detector = Mock()
    
    pipeline = ObjectDetectionPipeline()
    pipeline.camera_flow = mock_camera
    pipeline.detection_flow = mock_detector
    
    # Test execution
    result = pipeline.run(None)
    
    assert mock_camera.run_timed.called
    assert mock_detector.run_timed.called
```

## Performance Considerations

### Frame Rate Management
Different components may run at different rates:

```python
# Camera at 30Hz, detection at 10Hz, tracking at 15Hz
@flow(rate="30hz")
class HighFrequencyCapture(Flow): ...

@flow(rate="10hz") 
class MediumFrequencyDetection(Flow): ...

@flow(rate="15hz")
class MediumFrequencyTracking(Flow): ...
```

### Memory Management
Be conscious of memory usage in long-running pipelines:

```python
class MemoryEfficientPipeline(Flow[Input, Output]):
    def run_timed(self, input_data: Input, timer: ExecutionTimer) -> Output:
        # Process in chunks to limit memory usage
        results = []
        for chunk in self._chunk_input(input_data):
            chunk_result = self._process_chunk(chunk)
            results.append(chunk_result)
            # Clean up intermediate data
            del chunk
        
        return self._combine_results(results)
```

## Contributing

When creating new pipelines:

1. Identify common flow combinations
2. Design for reusability and configuration
3. Include comprehensive error handling
4. Add performance optimizations where appropriate
5. Write tests and documentation
6. Update category `__init__.py` exports
7. Consider FRP integration patterns

## Integration with Flows

Pipelines depend on flows from `../flows/`. Always prefer composing existing flows over reimplementing functionality. If you need new functionality, consider whether it belongs in a flow (atomic operation) or pipeline (composition of operations).