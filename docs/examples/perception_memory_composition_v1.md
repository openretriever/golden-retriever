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

## 5. Compose belief into downstream control

Once the belief stage is stable, compose it into a larger pipeline:

```bash
pixi run demo-perception-belief-control
```

This uses the staged-builder pattern from `examples/advanced/functional_wiring/`:
- build a perception slice
- surface the belief flow as the next stage boundary
- attach a downstream control slice explicitly

## 6. Where to go next

If you want more depth after this progression:
- `examples/advanced/perception_debug/README.md` for more perception-first debugging workflows
- `examples/advanced/state_management/README.md` for memory/state patterns
- `examples/advanced/functional_wiring/README.md` for composition patterns
- `examples/advanced/tamp_tabletop_pick_place/README.md` for a larger integrated planning/execution example
