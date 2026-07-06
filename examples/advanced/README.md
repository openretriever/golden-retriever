# Advanced Examples

This folder collects runnable GoldenRetriever examples that build on top of the core `retriever` runtime. Treat it as the source-checkout companion to the public Golden docs, not as a separate runtime package.

## Start Here

If you are new to Retriever, run the core visual quickstart first from `openretriever/retriever`. Then return here and prove that Golden extends the runtime through Retriever Hub before entering the applied example ladder.

```bash
# In openretriever/retriever first:
pixi run demo-webcam-detection-mock

# In openretriever/golden-retriever next:
pixi run demo-golden-hub-pack
pixi run -e golden demo-perception-detection-flow
pixi run demo-pipeline-html-viz
```

The first Golden command loads the current manifest-declared payload pack and Arrow helpers through Retriever Hub. The second starts the concise applied ladder. The third writes a graph artifact so users can see the pipeline boundary instead of only reading code.

Public guides for this folder live in the [Golden example catalog](../../docs/examples/README.md). Agent-oriented routing lives in [`docs/llms.txt`](../../docs/llms.txt). Keep new source examples out of the first-run path until they have a named Pixi task, expected output, and dependency level.

## Concise Golden Ladder

After the Hub proof is green, run the maintained mock-safe ladder:

```bash
pixi run -e golden demo-perception-detection-flow
pixi run -e golden demo-perception-segmentation-flow
pixi run -e golden demo-perception-pointing-flow
pixi run -e golden demo-memory-belief-flow
pixi run -e golden demo-memory-dropout-flow
pixi run -e golden demo-memory-pointing-flow
pixi run demo-perception-record
pixi run demo-perception-replay
pixi run -e golden demo-detection-window-stats
pixi run demo-perception-replay-to-belief
pixi run demo-perception-belief-control
pixi run -e golden demo-composable-pipelines
```

The concise perception, memory, language, and composition ladders use `golden` or `golden-perception` when they need the Golden example feature set.

## Recommended progression

1. `perception_examples/`: concise detection, segmentation, and pointing flows over one shared synthetic scene.
2. `memory_examples/`: concise belief, dropout-memory, and remembered-pointing flows built on the same perception payloads.
3. `language_examples/`: concise caption, grounding, and primitive plan-text flows over canonical core types.
4. `perception_debug/`: deterministic record/replay and windowed stats once the basic perception surfaces are clear.
5. `state_management/`: reset behavior, older belief examples, and event-driven replanning.
6. `functional_wiring/`: composing surfaced flows into larger pipelines.
7. `core_composition/`: registry-backed pipeline composition.
8. `multi_agent_communication/`: compact coordination/composition patterns.

Across these examples, prefer shared basic payloads plus structural composition. If a local stage needs grouped inputs or outputs, use composite `Flow[...]` typing first; only introduce a new named `Input` / `Output` envelope when that grouped boundary is itself a stable domain contract.

## Design notes

- `closed_loop_planning/`: extracted belief, monitoring, and replanning patterns from an older prototype. Keep it out of the first-run path because it is not a runnable example family.


## Optional Visualization and Simulator Lanes

Use these after the concise Golden ladder is clear. They are useful robot-facing references, but they are not the first-run contract because some depend on camera, simulator, viewer, or platform-specific packages.

```bash
pixi run demo-pipeline-html-viz
pixi run demo-robosuite-mock
pixi run -e torch demo-webcam-rerun
pixi run -e twist2 demo-twist2-rerun
```

- `webcam_rerun/`: webcam or mock perception with Rerun visualization and record/replay helpers.
- `twist2_simulation/`: MuJoCo/TWIST2 simulator integration with Rerun and optional native viewer.
- `robosuite_lift/`: mock-safe robosuite Lift smoke path plus optional real robosuite mode.
- `mujoco_manipulation/`: MuJoCo manipulation with Rerun logging.
- `hierarchical_physics_demo/`: Rerun plus HTML pipeline visualization for physics demos.
- `../experimental/visualization/`: deterministic IR/HTML pipeline visualization utility.

## Best entry points by topic

- `perception_examples/README.md`: the shortest path through detection, segmentation, and pointing.
- `memory_examples/README.md`: the shortest path through belief and remembered actions.
- `language_examples/README.md`: the shortest path through captions, grounding, and primitive plan text.
- `perception_debug/README.md`: stepper-first perception debugging and record/replay.
- `state_management/README.md`: older state, reset, and belief-update examples.
- `functional_wiring/README.md`: composition, fan-in/fan-out, and surfaced builders.
- `core_composition/README.md`: registry-backed pipeline composition surfaces (`pixi run -e golden demo-composable-pipelines`).
- `closed_loop_planning/README.md`: extracted belief, monitoring, and replanning patterns from the old prototype.
- `robosuite_lift/README.md`: mock-safe robosuite smoke demo and optional real-mode setup.
- `robotics_typing_standard/README.md`: typed payload and data-spec demos.
- `../../notebooks/README.md` (repo root): git-friendly notebook workflow for a small mechanics demo; keep the main runnable progression in the advanced example families above.

## Integrated walkthrough

For the public example-guide front door, start with `docs/examples/README.md`. For one self-contained article covering synthetic perception -> replay -> belief/memory -> composed control, continue with:

- `docs/examples/perception_and_memory_v1.md`
