# Perception Examples

These are concise advanced examples for the common perception stages you usually want to explain first:

1. `detection_flow.py`: detect simple objects from a deterministic scene.
2. `segmentation_flow.py`: turn the same scene into segmentation summaries.
3. `pointing_flow.py`: choose a normalized point target from detections.

All three examples now use the canonical primitive payloads from `retriever.types.perception`. The example-local code only keeps the deterministic scene logic and printers.

Run the examples through the Golden demo environment:

```bash
pixi run -e golden-local demo-perception-detection-flow
pixi run -e golden-local demo-perception-segmentation-flow
pixi run -e golden-local demo-perception-pointing-flow
```
