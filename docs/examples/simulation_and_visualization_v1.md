# Simulation and Visualization Examples

<div class="gr-route-pills gr-route-pills-inline">
  <a href="https://openretriever.org/">Retriever home</a>
  <a href="https://openretriever-docs.pages.dev/">Core docs</a>
  <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Visual quickstart</a>
  <a href="https://github.com/openretriever/retriever">Core source</a>
  <a href="/">Golden overview</a>
  <a href="https://github.com/openretriever/golden-retriever">Golden source</a>
  <a href="../llms.txt">Golden agent map</a>
</div>


These examples are the current Golden paths for visual and simulator-oriented demos. They are optional lanes after the concise perception -> memory -> language ladder.

## Quick Map

| Lane | Command | What it shows |
| --- | --- | --- |
| Webcam + Rerun | `pixi run -e torch demo-webcam-rerun` | Webcam/mock perception, model outputs, Rerun logging, record/replay helpers. |
| TWIST2 / MuJoCo | `pixi run -e twist2 demo-twist2-rerun` | Multi-rate simulator/policy/visualization loop. |
| MuJoCo Manipulation | Source-only: `pixi run -e twist2 python examples/advanced/mujoco_manipulation/app.py` | High-rate physics, slower controller, Rerun state logging. Not a promoted named task yet. |
| RoboSuite Lift | `pixi run demo-robosuite-mock` | Mock-safe robosuite wrapper and scripted Lift-policy contract. |
| Hierarchical Physics | Source-only: `pixi run -e twist2 python examples/advanced/hierarchical_physics_demo/app.py --demo both --duration 8` | Explicit clock -> sim -> viz layers plus HTML pipeline visualization. Not a promoted named task yet. |

## Safe First Visual Checks

Start with the two commands that do not require a camera, robot, model key, MuJoCo, TWIST2, or robosuite install:

```bash
pixi run demo-robosuite-mock
pixi run demo-pipeline-html-viz
```

Expected results:

- `demo-robosuite-mock` prints `[mock step=...]` lines showing object height, gripper height, reward, and done state.
- `demo-pipeline-html-viz` prints an ASCII graph and writes `out/golden_retriever_closed_loop_viz.html`.

Use the richer Rerun and simulator lanes after these pass on the same checkout.

## Webcam + Rerun

```bash
pixi run -e torch demo-webcam-rerun
```

What it shows:

- webcam or mock image input,
- model-backed open-vocabulary detection/segmentation,
- Rerun visualization,
- record/replay helpers for MCAP-oriented debugging.

Source:

- `examples/advanced/webcam_rerun/README.md`
- `examples/advanced/webcam_rerun/app.py`

## TWIST2 / MuJoCo

```bash
pixi run -e twist2 demo-twist2-rerun
```

For native MuJoCo viewer support on machines configured for it:

```bash
pixi run -e twist2 demo-twist2
```

What it shows:

- MuJoCo physics and policy loops running at different rates,
- `@gui_flow` for main-thread visualization,
- Rerun logging for headless inspection,
- a concrete simulator integration built around Retriever Flows.

Source:

- `examples/advanced/twist2_simulation/README.md`
- `examples/advanced/twist2_simulation/app.py`

## MuJoCo Manipulation Source Reference

This lane has a concrete launch file, but it is source-only until it receives a named Pixi task and public smoke coverage. Use it after the promoted visual checks pass.

```bash
pixi run -e twist2 python examples/advanced/mujoco_manipulation/app.py
```

What it shows:

- high-rate MuJoCo physics,
- slower controller Flow,
- Rerun visualization of camera/render output, target/tip geometry, and joint state plots,
- explicit synchronization between physics, control, and visualization rates.

Source:

- `examples/advanced/mujoco_manipulation/README.md`
- `examples/advanced/mujoco_manipulation/app.py`

## RoboSuite Lift Smoke Demo

```bash
pixi run demo-robosuite-mock
```

This is the default smoke path: it exercises the Retriever graph contract without requiring robosuite. For a real robosuite `Lift` run, install the optional dependency and run the real mode:

```bash
# --no-deps: base deps resolve from the pixi env; retriever-core is not on
# PyPI until the core runtime publishes (drop the flag after that release).
pixi run python -m pip install --no-deps -e ".[robosuite]" robosuite
pixi run demo-robosuite-lift
```

What it shows:

- simulator/environment wrapper and policy as separate Flows,
- slow policy updates coupled to faster simulator state with `Latest()`,
- a mock path for CI/docs and a real robosuite path for configured machines.

Source:

- `examples/advanced/robosuite_lift/README.md`
- `examples/advanced/robosuite_lift/app.py`

## Hierarchical Physics Source Reference

This lane is useful for design inspection, but it is source-only until it receives a named Pixi task and stable public smoke coverage. The promoted lightweight graph artifact remains `pixi run demo-pipeline-html-viz`.

```bash
pixi run -e twist2 python examples/advanced/hierarchical_physics_demo/app.py --demo both --duration 8
```

What it shows:

- explicit clock -> simulation -> visualization layers,
- Rerun logging for physics state,
- generated interactive HTML pipeline visualization,
- ASCII pipeline summaries for quick inspection.

Source:

- `examples/advanced/hierarchical_physics_demo/README.md`
- `examples/advanced/hierarchical_physics_demo/app.py`

## Self-Contained Pipeline HTML Viz

Use this when you want a small deterministic graph artifact without simulator, camera, robot, or model dependencies.

```bash
pixi run demo-pipeline-html-viz
```

What it shows:

- a small cyclic dummy pipeline,
- IR generation,
- ASCII graph output,
- HTML export written to `out/golden_retriever_closed_loop_viz.html`.

This is intentionally lightweight: a graph-visualization utility rather than a full simulator example family.

## Current Scope

This page only lists examples that exist in the current repository and have concrete launch files. Keep new simulator lanes out of the public front door until they have a runnable command, dependency story, and README.
