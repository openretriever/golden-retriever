# Advanced Examples

This folder collects runnable GoldenRetriever demos that build on top of the core `retriever` runtime.

## Start Here

```bash
pixi run demo-synthetic-color-stepper
pixi run demo-perception-record
pixi run demo-perception-replay
pixi run demo-stateful-reset
pixi run demo-belief-updater-internal
pixi run demo-belief-updater-explicit
pixi run demo-perception-replay-to-belief
pixi run demo-perception-belief-control
```

## Recommended progression

1. `perception_debug/`: synthetic perception and record/replay without hardware dependencies.
2. `state_management/`: reset behavior, internal memory, and explicit-state belief updates.
3. `functional_wiring/`: composing surfaced flows into larger pipelines.
4. `multi_agent_communication/`: compact coordination/composition patterns.
5. `tamp_tabletop_pick_place/`: a larger integrated planning + execution demo with a simulator.

## Best entry points by topic

- `perception_debug/README.md`: stepper-first perception debugging.
- `state_management/README.md`: state, reset, and belief updates.
- `functional_wiring/README.md`: composition, fan-in/fan-out, and surfaced builders.
- `robotics_typing_standard/README.md`: typed payload and data-spec demos.
- `notebooks/README.md`: git-friendly notebook workflow for a small mechanics demo; keep the main runnable progression in the advanced example families above.

## Integrated walkthrough

For one self-contained article covering synthetic perception -> replay -> belief/memory -> composed control, see:

- `docs/examples/perception_memory_composition_v1.md`
