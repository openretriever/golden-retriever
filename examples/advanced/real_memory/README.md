# Real Memory

These examples reuse the same `SceneBelief` and `PointTarget2D` contracts from `memory_examples/`, but feed them from explicit real/mock perception backends.

## Start Here

```bash
pixi run -e golden-perception demo-belief-from-real-detections
pixi run -e golden-perception demo-grounded-reference-memory
```

## Design rule

Real memory stays small on purpose:
- reuse `DetectionBatch` from the perception side
- reuse `BeliefTracker` from the memory side
- keep backend choice explicit (`mock` or `gemini_api`)

This keeps the real-memory path aligned with the synthetic ladder instead of creating another example-only boundary type.
