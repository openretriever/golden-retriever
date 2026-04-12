# Perception, Memory, and Composition v1

This guide walks one concrete GoldenRetriever progression from a minimal synthetic perception loop to a composed perception -> belief -> control pipeline.

## 1. Start with deterministic synthetic perception

Use the smallest perception loop first so debugging stays local and reproducible.

```bash
pixi run demo-synthetic-color-stepper
```

What to look for:
- a synthetic image source instead of a live camera
- deterministic stepping and inspectable outputs
- a minimal detector path before any replay or memory is introduced

## 2. Record one short perception session and replay it

Record a short synthetic session to MCAP, then replay it without re-running the source.

```bash
pixi run demo-perception-record
pixi run demo-perception-replay
```

This gives you a stable artifact you can feed into later stages.

## 3. Move from raw perception into memory / belief state

Start with the smallest stateful examples:

```bash
pixi run demo-stateful-reset
pixi run demo-belief-updater-internal
pixi run demo-belief-updater-explicit
```

Use these to answer two different questions:
- what exactly does `pipe.reset()` clear?
- should state live inside one flow, or be passed explicitly through the graph?

## 4. Feed replayed perception into a belief updater

Now bridge the replay artifact into a memory-bearing stage:

```bash
pixi run demo-perception-replay-to-belief
```

This is the most direct perception -> memory handoff in the repo:
- replayed detections become the stable input surface
- the belief stage accumulates state across steps
- you can inspect the pipeline without requiring live sensors

The intended design rule here is shared payloads first: replayed detections flow into one belief-state payload, and downstream stages consume that same stable shape instead of defining one-off IO envelope classes per node.

## 5. Compose belief into downstream control

Once the belief stage is stable, compose it into a larger pipeline:

```bash
pixi run demo-perception-belief-control
```

This uses the staged-builder pattern from `examples/advanced/functional_wiring/`:
- build a perception slice
- surface the belief flow as the next stage boundary
- attach a downstream control slice explicitly

The composition is structural: stages are wired together around a small shared payload vocabulary, not around pipeline-specific wrapper dataclasses.

## 6. Add one more perception surface: windowed stats

If you want one more perception-side debugging surface before moving on, run:

```bash
pixi run -e golden-local demo-detection-window-stats
```

This keeps the same deterministic synthetic camera source, but adds a windowed aggregation stage so you can see how temporal statistics sit between raw detections and downstream memory. Right now it expects the local editable-core env because the bundled wheel in `retriever_dist` still uses the older startup semantics.

## 7. Add one more memory surface: stateful replanning

To see internal planner memory without bringing in a full robot stack, run:

```bash
pixi run -e golden-local demo-stateful-replanning
```

This example keeps state inside the replanner and emits plan updates only when obstacle events occur or clear. It also currently expects the local editable-core env so the runtime startup path matches the reset-first contract.

## 8. Next: newer core composition surfaces

To explore the newer registry-backed composition surfaces from the current `retriever-mirror` core, switch to the local editable-core env and run:

```bash
pixi install -e golden-local
pixi run -e golden-local demo-composable-pipelines
```

That example demonstrates:
- surfaced input injection into a named internal stage
- replacing an internal stage after pipeline construction
- wrapping a registered pipeline back into a larger graph via `build_pipeline_flow(...)`

Again, the point is to keep payloads stable while changing structure around them.

For a dedicated walkthrough of that surface, continue with `docs/examples/core_composition_surfaces_v1.md`.
