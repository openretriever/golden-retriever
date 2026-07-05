# Perception, Memory, and Composition v1

<div class="gr-route-pills gr-route-pills-inline">
  <a href="https://openretriever.org/">Retriever home</a>
  <a href="https://openretriever.org/start/">Start path</a>
  <a href="https://openretriever-docs.pages.dev/">Core docs</a>
  <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Visual quickstart</a>
  <a href="/">Golden overview</a>
  <a href="../llms.txt">Golden agent map</a>
</div>


This guide walks one concrete GoldenRetriever progression from a minimal synthetic perception loop to a composed perception -> belief -> control pipeline.

## 1. Start with the concise perception ladder

Use the smallest perception flows first so debugging stays local and reproducible.

```bash
pixi run -e golden-local demo-perception-detection-flow
pixi run -e golden-local demo-perception-segmentation-flow
pixi run -e golden-local demo-perception-pointing-flow
```

What to look for:
- one deterministic camera surface reused across all three examples
- detection, segmentation, and pointing built from `retriever.types.perception` primitives plus local flow logic
- no one-off `Input` / `Output` shells just to move between stages

## 2. Add the concise memory ladder

Then add the smallest memory-bearing flows on top of the same perception payloads:

```bash
pixi run -e golden-local demo-memory-belief-flow
pixi run -e golden-local demo-memory-dropout-flow
pixi run -e golden-local demo-memory-pointing-flow
```

These show the intended composition rule directly:
- perception emits stable payloads
- memory layers consume the same payloads
- later stages change structure around those payloads instead of redefining them

Composite flow typing is usually enough here. If one local stage needs `(Image2D, DetectionBatch)` or `(SceneBelief, GoalSpec)`, prefer `Flow[(A, B), C]` over inventing another example-only `Input` dataclass.

## 3. Record one short perception session and replay it

Record a short synthetic session to MCAP, then replay it without re-running the source.

```bash
pixi run demo-perception-record
pixi run demo-perception-replay
```

This gives you a stable artifact you can feed into later stages.

## 4. Compare with the older state-management surfaces

Start with the smallest stateful examples:

```bash
pixi run demo-stateful-reset
pixi run demo-belief-updater-internal
pixi run demo-belief-updater-explicit
```

Use these to answer two different questions:
- what exactly does `pipe.reset()` clear?
- should state live inside one flow, or be passed explicitly through the graph?

## 5. Feed replayed perception into a belief updater

Now bridge the replay artifact into a memory-bearing stage:

```bash
pixi run demo-perception-replay-to-belief
```

This is the most direct perception -> memory handoff in the repo:
- replayed detections become the stable input surface
- the belief stage accumulates state across steps
- you can inspect the pipeline without requiring live sensors

The intended design rule here is shared payloads first: replayed detections flow into one belief-state payload, and downstream stages consume that same stable shape instead of defining one-off IO envelope classes per node. In this ladder, the local belief carriers live in `examples/advanced/memory_examples/types.py` rather than inside the flow helper module.

## 6. Compose belief into downstream control

Once the belief stage is stable, compose it into a larger pipeline:

```bash
pixi run demo-perception-belief-control
```

This uses the staged-builder pattern from `examples/advanced/functional_wiring/`:
- build a perception slice
- surface the belief flow as the next stage boundary
- attach a downstream control slice explicitly

The composition is structural: stages are wired together around a small shared payload vocabulary, not around pipeline-specific wrapper dataclasses.

## 7. Add one more perception surface: windowed stats

If you want one more perception-side debugging surface before moving on, run:

```bash
pixi run -e golden-local demo-detection-window-stats
```

This keeps the same deterministic synthetic camera source, but adds a windowed aggregation stage so you can see how temporal statistics sit between raw detections and downstream memory.

## 8. Add one more memory surface: stateful replanning

To see internal planner memory without bringing in a full robot stack, run:

```bash
pixi run -e golden-local demo-stateful-replanning
```

This example keeps state inside the replanner and emits plan updates only when obstacle events occur or clear.

## 9. Next: newer core composition surfaces

To explore the newer registry-backed composition surfaces, run:

```bash
pixi run -e golden-local demo-composable-pipelines
```

That example demonstrates:
- surfaced input injection into a named internal stage
- replacing an internal stage after pipeline construction
- wrapping a registered pipeline back into a larger graph via `build_pipeline_flow(...)`

Again, the point is to keep payloads stable while changing structure around them.

## 10. Add one small language and grounding ladder

If you want the smallest text-facing examples without jumping into model-specific packets, continue with:

```bash
pixi run -e golden-local demo-language-caption-plan
pixi run -e golden-local demo-language-grounded-reference
```

These use the canonical core language primitives directly and keep structural composition explicit.

For the dedicated walkthrough, continue with [Language and Grounding](language_and_grounding_v1.md).

## 11. Optional: explicit real-model backends

Once the concise synthetic ladders are clear, you can switch to explicit real/mock backends that keep the same payload contracts:

```bash
pixi run -e golden-perception demo-gemini-detection-flow
pixi run -e golden-perception demo-gemini-pointing-flow
pixi run -e golden-perception demo-belief-from-real-detections
pixi run -e golden-perception demo-grounded-reference-memory
```

These examples stay secondary on purpose: they are useful for model integration and grounded-reference experiments, but the main teaching path should remain deterministic and easy to inspect.
