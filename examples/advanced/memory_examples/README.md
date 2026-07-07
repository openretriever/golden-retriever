# Memory Examples

These are concise advanced examples for the common memory and belief stages that follow perception:

1. `belief_from_detections.py`: build a small scene belief from detections.
2. `memory_under_dropout.py`: keep belief state stable when detections disappear.
3. `pointing_with_memory.py`: keep target pointing stable through intermittent perception.

All three examples reuse the canonical detection and pointing payloads from `retriever.types.perception`. The local memory-only state payloads live in `memory_examples/types.py`, so the flows stay separate from the belief carriers they manipulate.

Run the examples through the Golden demo environment:

```bash
pixi run -e golden-retriever demo-memory-belief-flow
pixi run -e golden-retriever demo-memory-dropout-flow
pixi run -e golden-retriever demo-memory-pointing-flow
```
