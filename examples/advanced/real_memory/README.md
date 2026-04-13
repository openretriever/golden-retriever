# Real Memory

These examples reuse the local `SceneBelief` memory state from `examples/advanced/memory_examples/types.py` together with canonical `PointTarget2D` and `DetectionBatch` payloads from `retriever.types.perception`, but feed them from explicit real/mock perception backends.

## Start Here

```bash
pixi run -e golden-perception demo-belief-from-real-detections
pixi run -e golden-perception demo-grounded-reference-memory
```

Start with the mock tasks first. They keep the same surface but avoid credentials and local model setup.

## Design rule

Real memory stays small on purpose:
- reuse canonical `DetectionBatch` from `retriever.types.perception`
- reuse `BeliefTracker` from the memory side
- keep backend choice explicit (`mock` or `gemini_api`)

This keeps the real-memory path aligned with the synthetic ladder instead of creating another example-only boundary type.
