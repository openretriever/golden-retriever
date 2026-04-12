# Memory Examples

These are concise advanced examples for the common memory and belief stages that follow perception:

1. `belief_from_detections.py`: build a small scene belief from detections.
2. `memory_under_dropout.py`: keep belief state stable when detections disappear.
3. `pointing_with_memory.py`: keep target pointing stable through intermittent perception.

All three examples reuse the detection payloads from `examples.advanced.perception_examples.common`. Memory stays layered on top of the perception primitives instead of redefining frame or detection carriers.

Run:

```bash
pixi run demo-memory-belief-flow
pixi run demo-memory-dropout-flow
pixi run demo-memory-pointing-flow
```
