# Retriever Flows

This directory contains reusable flow components for the Retriever robotics framework. Flows are atomic, composable building blocks that implement the `Flow[I, O]` interface for type-safe pipeline construction.

## Organization

### Vision Flows (`vision/`)
Computer vision and perception components:
- **camera.py**: Image capture and camera interfaces
- **detection.py**: Object detection models and algorithms  
- **depth.py**: Depth estimation and 3D perception
- **segmentation.py**: Pixel-level image understanding

### Control Flows (`control/`)
Robot actuation and movement components:
- **arm.py**: Robotic arm control and manipulation
- **navigation.py**: Mobile base movement and pathfinding
- **gripper.py**: End-effector and grasping control
- **safety.py**: Safety monitoring and constraint checking

### Reasoning Flows (`reasoning/`)
Planning and decision-making components:
- **planning.py**: Task and motion planning algorithms
- **learning.py**: Skill acquisition and adaptation
- **symbolic.py**: Logic and symbolic reasoning
- **decision.py**: Decision-making under uncertainty

### Sensing Flows (`sensing/`)
Sensor processing and data acquisition:
- **sensors.py**: Raw sensor data processing
- **fusion.py**: Multi-sensor data integration
- **processing.py**: Signal processing and filtering
- **calibration.py**: Sensor calibration and management

## Usage Examples

### Basic Flow Usage

```python
from retriever.flows.vision.camera import CameraFlow
from retriever.flows.vision.detection import YOLOFlow

# Create flows
camera = CameraFlow(camera_id=0)
detector = YOLOFlow()

# Execute individually
with camera:
    image = camera.run(None)
    detections = detector.run(image)
```

### Flow Composition

```python
from retriever.core.types import Pipeline

# Compose flows using >> operator
pipeline = camera >> detector >> filter_flow

# Execute composed pipeline
results = pipeline.execute(None)
```

### FRP Integration

```python
from retriever.core.frp import flow

@flow(rate="30hz")
class CustomFlow(Flow[Input, Output]):
    def run_timed(self, data: Input, timer: ExecutionTimer) -> Output:
        # Your processing logic here
        return result
```

## Flow Development Guidelines

### 1. Type Safety
All flows must specify clear input/output types:

```python
class MyFlow(Flow[InputType, OutputType]):
    def run_timed(self, input_data: InputType, timer: ExecutionTimer) -> OutputType:
        # Implementation
        pass
```

### 2. FRP Annotations
Use `@flow()` decorator for reactive flows:

```python
@flow(rate="10hz", trigger="data_available")
class ReactiveFlow(Flow[Input, Output]):
    # Implementation
```

### 3. Error Handling
Implement robust error handling:

```python
def run_timed(self, input_data: Input, timer: ExecutionTimer) -> Output:
    try:
        # Processing logic
        return result
    except Exception as e:
        logger.error(f"Flow {self.__class__.__name__} failed: {e}")
        raise
```

### 4. Resource Management
Use context managers for resource cleanup:

```python
class ResourceFlow(Flow[Input, Output]):
    def __enter__(self):
        self.resource = acquire_resource()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.resource.release()
```

### 5. Configuration
Support flexible configuration:

```python
class ConfigurableFlow(Flow[Input, Output]):
    def __init__(self, param1: float = 0.5, param2: str = "default"):
        self.param1 = param1
        self.param2 = param2
```

## Testing

Each flow should include comprehensive tests:

```python
import pytest
from retriever.flows.vision.detection import YOLOFlow

def test_yolo_flow():
    flow = YOLOFlow()
    # Test implementation
    assert flow is not None
```

## Contributing

When adding new flows:

1. Place in appropriate category directory
2. Follow naming conventions (`*Flow` suffix)
3. Include comprehensive docstrings
4. Add to category `__init__.py` exports
5. Create tests in `tests/flows/`
6. Update this README if adding new categories

## Integration with Pipelines

Flows are designed to work seamlessly with the pipelines in `../pipelines/`. Pipelines compose multiple flows to accomplish complex tasks, while flows remain focused on single responsibilities.

See `../pipelines/README.md` for pipeline development guidelines.