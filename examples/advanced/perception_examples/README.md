# Perception Examples

These are concise advanced examples for the common perception stages you usually want to explain first:

1. `detection_flow.py`: detect simple objects from a deterministic scene.
2. `segmentation_flow.py`: turn the same scene into segmentation summaries.
3. `pointing_flow.py`: choose a normalized point target from detections.

All three examples reuse the same small payload vocabulary from `common.py`.

Run:

```bash
pixi run demo-perception-detection-flow
pixi run demo-perception-segmentation-flow
pixi run demo-perception-pointing-flow
```
