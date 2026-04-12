# Advanced Examples

This folder collects runnable GoldenRetriever demos that build on top of the core `retriever` runtime.

## Start Here

```bash
pixi run demo-perception-detection-flow
pixi run demo-perception-segmentation-flow
pixi run demo-perception-pointing-flow
pixi run demo-memory-belief-flow
pixi run demo-memory-dropout-flow
pixi run demo-memory-pointing-flow
pixi run demo-perception-record
pixi run demo-perception-replay
pixi run -e golden-local demo-detection-window-stats
pixi run demo-perception-replay-to-belief
pixi run demo-perception-belief-control
pixi run -e golden-local demo-composable-pipelines
```

The `golden-local` launch points above require the local editable-core environment; the rest run on the bundled default setup.

## Recommended progression

1. `perception_examples/`: concise detection, segmentation, and pointing flows over one shared synthetic scene.
2. `memory_examples/`: concise belief, dropout-memory, and remembered-pointing flows built on the same perception payloads.
3. `perception_debug/`: deterministic record/replay and windowed stats once the basic perception surfaces are clear.
4. `state_management/`: reset behavior, older belief examples, and event-driven replanning.
5. `functional_wiring/`: composing surfaced flows into larger pipelines.
6. `core_composition/`: registry-backed pipeline composition using a local editable core checkout.
7. `multi_agent_communication/`: compact coordination/composition patterns.
8. `tamp_tabletop_pick_place/`: a larger integrated planning + execution demo with a simulator.

Across these examples, prefer shared basic payloads (`tuple[...]`, belief/state dataclasses, shared symbolic actions/atoms, stamped spatial types) plus structural composition. Do not treat each stage as a reason to invent a new `Input` / `Output` envelope unless the grouped shape is itself a stable domain contract.

## Best entry points by topic

- `perception_examples/README.md`: the shortest path through detection, segmentation, and pointing.
- `memory_examples/README.md`: the shortest path through belief and remembered actions.
- `perception_debug/README.md`: stepper-first perception debugging and record/replay.
- `state_management/README.md`: older state, reset, and belief-update examples.
- `functional_wiring/README.md`: composition, fan-in/fan-out, and surfaced builders.
- `core_composition/README.md`: registry-backed pipeline composition surfaces (`pixi run -e golden-local ...`).
- `robotics_typing_standard/README.md`: typed payload and data-spec demos.
- `notebooks/README.md`: git-friendly notebook workflow for a small mechanics demo; keep the main runnable progression in the advanced example families above.

## Integrated walkthrough

For the public example-guide front door, start with `docs/examples/README.md`. For one self-contained article covering synthetic perception -> replay -> belief/memory -> composed control, continue with:

- `docs/examples/perception_and_memory_v1.md`
