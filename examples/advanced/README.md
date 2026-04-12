# Advanced Examples

This folder collects runnable GoldenRetriever demos that build on top of the core `retriever` runtime.

## Start Here

```bash
pixi run demo-synthetic-color-stepper
pixi run demo-perception-record
pixi run demo-perception-replay
pixi run -e golden-local demo-detection-window-stats
pixi run demo-stateful-reset
pixi run demo-belief-updater-internal
pixi run demo-belief-updater-explicit
pixi run -e golden-local demo-stateful-replanning
pixi run demo-perception-replay-to-belief
pixi run demo-perception-belief-control
pixi run -e golden-local demo-composable-pipelines
```

The three `golden-local` launch points above require the local editable-core environment; the rest run on the bundled default setup.

## Recommended progression

1. `perception_debug/`: synthetic perception, windowed stats, and record/replay without hardware dependencies.
2. `state_management/`: reset behavior, internal memory, explicit-state belief updates, and event-driven replanning.
3. `functional_wiring/`: composing surfaced flows into larger pipelines.
4. `core_composition/`: registry-backed pipeline composition using a local editable core checkout.
5. `multi_agent_communication/`: compact coordination/composition patterns.
6. `tamp_tabletop_pick_place/`: a larger integrated planning + execution demo with a simulator.

Across these examples, prefer shared basic payloads (`tuple[...]`, belief/state dataclasses, shared symbolic actions/atoms, stamped spatial types) plus structural composition. Do not treat each stage as a reason to invent a new `Input` / `Output` envelope unless the grouped shape is itself a stable domain contract.

## Best entry points by topic

- `perception_debug/README.md`: stepper-first perception debugging.
- `state_management/README.md`: state, reset, and belief updates.
- `functional_wiring/README.md`: composition, fan-in/fan-out, and surfaced builders.
- `core_composition/README.md`: registry-backed pipeline composition surfaces (`pixi run -e golden-local ...`).
- `robotics_typing_standard/README.md`: typed payload and data-spec demos.
- `notebooks/README.md`: git-friendly notebook workflow for a small mechanics demo; keep the main runnable progression in the advanced example families above.

## Integrated walkthrough

For one self-contained article covering synthetic perception -> replay -> belief/memory -> composed control, see:

- `docs/examples/perception_memory_composition_v1.md`
