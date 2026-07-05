# Pipeline Composition

Use this page when a single Flow is no longer enough and you want a reusable robot pipeline: perception into belief, belief into planning, planning into control, or any similar applied graph.

The core Retriever docs teach the runtime model. This Golden page shows how the same ideas appear in robot-facing examples that can be tested and reused.

## Run the Example

```bash
pixi install -e golden-local
pixi run -e golden-local demo-composable-pipelines
```

Expected result: the command builds a small registered pipeline, swaps one stage, wraps the pipeline as a Flow, and prints a compact summary. It should not require a robot, simulator, camera, or network service.

## What It Demonstrates

<div class="gr-fit-grid">
  <div class="gr-fit-card">
    <span>Register</span>
    <strong>Name a reusable pipeline</strong>
    <p>Give a multi-stage graph a stable name so examples and future Hub packs can refer to it without copying wiring code.</p>
  </div>
  <div class="gr-fit-card">
    <span>Swap</span>
    <strong>Replace an internal stage</strong>
    <p>Keep the public pipeline boundary fixed while changing one implementation detail, such as a detector, memory updater, or planner.</p>
  </div>
  <div class="gr-fit-card">
    <span>Wrap</span>
    <strong>Use a pipeline as a Flow</strong>
    <p>Embed a reusable pipeline inside a larger graph when it becomes one logical robot subsystem.</p>
  </div>
</div>

## When To Use This

Use explicit Flow wiring first when you are learning Retriever or debugging one graph. Use registered composition when a pipeline boundary is reused across examples, notebooks, or future Hub packs.

| Situation | Preferred surface |
| --- | --- |
| Learning how values move through a graph | Explicit Flow wiring |
| Debugging one example locally | Explicit Flow wiring plus graph visualization |
| Reusing the same subgraph across examples | Registered pipeline composition |
| Publishing a stable reusable boundary | Hub pack candidate after import/version/smoke checks |

## Source Pointer

The runnable example lives in `examples/advanced/core_composition/composable_pipelines.py`.

Related Golden examples:

- `examples/advanced/perception_debug/detection_window_stats.py`: temporal aggregation in a perception pipeline.
- `examples/advanced/state_management/stateful_replanning.py`: planner memory and change-only events.
- `examples/advanced/functional_wiring/perception_belief_control_pipeline.py`: explicit belief-to-control graph wiring.

## Notebook Path

```bash
pixi run notebook-to-ipynb-hub
pixi install -e golden-local
pixi run -e golden-local demo-hub-notebook-source
```

The notebook source is `notebooks/src/hub_demo.py`. It stays parameterized so public docs do not hardcode private or organization-specific Hub references.
