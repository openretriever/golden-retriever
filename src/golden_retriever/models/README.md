# Model Pipelines and Interfaces

## Overview
This document outlines the model pipelines and interfaces used in the project, with a focus on pointing and segmentation capabilities.

## Current Model Pipelines

### 1. Pointing + Segmentation Pipeline
Located in `src/models/segmentation/`:
- `pointing_gemini_sam2.py`: Main service combining Gemini for pointing and SAM2 for segmentation
- `pointing_gemini_sam2_standalone.py`: Standalone version of the service with HTTP server
- `pointing_gemini_sam2_client.py`: HTTP client for the Pointing Gemini SAM2 service
- Other variants:
  - `molmo_sam2_http_server.py` and `molmo_sam2_http_client.py`: HTTP-based serving
  - `molmo_sam2_ray_server.py` and `molmo_sam2_ray_client.py`: Ray-based serving
  - `molmo_sam2_lmdeploy_http_server.py`: LMDeploy-based serving

### 2. Model Inference Interfaces
Located in:
- `src/envs/alf_world/agents/agent_utils/inference_engine.py`: Base inference engine supporting multiple providers
- `src/models/api_models/utils/`: Provider-specific utilities
  - `openai_utils.py`: OpenAI API utilities
  - `google_utils.py`: Google API utilities
  - `anthropic_utils.py`: Anthropic API utilities

## Base Classes
Located in `src/models/model_base.py`:
- `ModelActorBase`: Base class for Ray-based model actors
- `BaseModelServer`: Base class for Ray Serve-based model servers
- `VisLangBase`: Base class for vision-language models

## Pointing Gemini SAM2 Service

### Overview
The Pointing Gemini SAM2 service combines Gemini's pointing capability with SAM2's segmentation capabilities. It provides a unified HTTP interface for:
- Pointing to objects in images using Gemini
- Detecting objects with bounding boxes using Gemini
- Segmenting objects using SAM2 (Segment Anything Model 2)
- Visualizing results
- Detailed timing and logging information

### Capabilities

#### 1. Pointing (Gemini)
The service can point to specific objects in images using natural language prompts. For example:
- "Point to the red cup on the table"
- "Show me where the laptop is"
- "Indicate the position of the chair"

The response includes:
- Point coordinates (x, y) in image space
- Labels for each point
- Visualization of points with labels
- Detailed timing information for each operation

#### 2. Detection (Gemini)
The service can detect and localize objects in images using bounding boxes. For example:
- "Detect all objects in the image"
- "Find all cups on the table"
- "Locate the furniture in the room"

The response includes:
- Bounding box coordinates [y1, x1, y2, x2] in image space
- Labels for each detected object
- Visualization of boxes with labels
- Detailed timing information for each operation

#### 3. Segmentation (SAM2)
The service can segment objects using SAM2 (Segment Anything Model 2), either from points or boxes. For example:
- "Segment the object at this point"
- "Segment all detected objects"
- "Create masks for the furniture"

The response includes:
- Binary masks for each object
- Labels for each mask
- Visualization of masks with labels
- Detailed timing information for each operation

Note: Segmentation is performed using SAM2, which requires:
- A valid point or box input
- GPU acceleration for optimal performance
- Proper initialization of the SAM2 model

TODO: Add support for Gemini-based segmentation when available in the API

### Usage Examples

#### 1. Starting the Server
```bash
# Main service
python -m src.models.segmentation.pointing_gemini_sam2 --port 7100 --use-gpu

# Standalone service
python -m src.models.segmentation.pointing_gemini_sam2_standalone --port 7100 --use-gpu
```

#### 2. Using the Client
```python
from src.models.segmentation.pointing_gemini_sam2_client import PointingGeminiSAM2Client

# Initialize client
client = PointingGeminiSAM2Client(host="localhost", port=7100)

# Get point predictions
result = client.predict(
    images=["image1.jpg", "image2.jpg"],
    prompts=["Point at the cups", "Point at the table"],
    points=True,
    segmentation=False,
    detection=False
)
print("Points:", result["results"])
print("Timing:", result["timings"])

# Get detection predictions
result = client.predict(
    images=["image1.jpg"],
    prompts=["Detect all objects in the image"],
    points=False,
    segmentation=False,
    detection=True
)
print("Detections:", result["results"])
print("Timing:", result["timings"])

# Get segmentation predictions (requires points or boxes)
result = client.predict(
    images=["image1.jpg"],
    prompts=["Segment the object at this point"],
    points=True,  # or detection=True for box-based segmentation
    segmentation=True,
    detection=False
)
print("Segmentation:", result["results"])
print("Timing:", result["timings"])

# TODO: Add example for Gemini-based segmentation when available
```

#### 3. Command Line Interface
```bash
# Pointing
python -m src.models.segmentation.pointing_gemini_sam2_client \
    --image image1.jpg \
    --prompt "Point at the cups" \
    --points --no-segmentation --no-detection

# Detection
python -m src.models.segmentation.pointing_gemini_sam2_client \
    --image image1.jpg \
    --prompt "Detect all objects" \
    --no-points --no-segmentation --detection

# Segmentation (requires points or boxes)
python -m src.models.segmentation.pointing_gemini_sam2_client \
    --image image1.jpg \
    --prompt "Segment the object at this point" \
    --points --segmentation --no-detection

# TODO: Add CLI example for Gemini-based segmentation when available
```

### Response Format
The service returns results in the following format:
```python
{
    "results": [
        {
            "image_index": 0,
            "prompt_index": 0,
            "prompt": "Point at the cups",
            "points": [
                {"point": [500, 500], "label": "cup1"},
                {"point": [300, 300], "label": "cup2"}
            ],
            "boxes": [
                [100, 100, 200, 200],
                [300, 300, 400, 400]
            ],
            "masks": [<PIL.Image.Image>, <PIL.Image.Image>],
            "image_width": 1000,
            "image_height": 1000,
            "normalization_info": {
                "width": 1000,
                "height": 1000,
                "normalized_width": 1000,
                "normalized_height": 1000
            }
        }
    ],
    "timings": {
        "total_time": 1.234,
        "per_image_times": [0.567, 0.667],
        "per_prompt_times": [0.123, 0.234, 0.345, 0.456],
        "avg_time_per_image": 0.617,
        "avg_time_per_prompt": 0.308
    }
}
```

### Features
- Support for multiple images and prompts
- Async and sync client interfaces
- Progress bars and timing measurements
- GPU acceleration support (required for SAM2)
- Error handling and user feedback
- Visualization of points, boxes, and masks
- Label support for all detections
- Detailed timing information for:
  - Total processing time
  - Per-image processing times
  - Per-prompt processing times
  - Average times for images and prompts
- Verbose logging with formatted output
- Visualization saving with customizable parameters

## Model Access Patterns

### 1. Direct API Calls
Using provider-specific utilities in `api_models/utils/`:
```python
from src.models.api_models.utils.google_utils import GeminiClient
client = GeminiClient()
response = client.generate(...)
```

### 2. Inference Engine
Using the unified inference engine:
```python
from src.envs.alf_world.agents.agent_utils.inference_engine import engine_factory
engine = engine_factory("gemini-1.5-pro")
response = engine.generate(...)
```

### 3. HTTP-based Serving
Using the HTTP client:
```python
client = PointingGeminiSAM2Client(host="localhost", port=7100)
result = client.predict(image, prompt)
```

## Future Improvements
- Standardize model interfaces across providers
- Add support for more providers
- Improve error handling and retry logic
- Add comprehensive testing
- Document best practices for model deployment
- Add support for video input
- Improve visualization capabilities
- Add support for more complex queries
- Optimize SAM2 performance
- Add support for different SAM2 variants
- TODO: Add Gemini-based segmentation support when available
- TODO: Add support for Gemini's multimodal capabilities