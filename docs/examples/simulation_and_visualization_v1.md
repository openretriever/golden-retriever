# Simulation and Visualization Examples


These examples are the Golden paths for visual and simulator-oriented demos. Start with the mock-safe checks, then move into webcam, Rerun, MuJoCo, TWIST2, or robosuite only when those dependencies are available.

## Quick Map

| Lane | First command | Status | What it shows |
| --- | --- | --- | --- |
| Webcam + Rerun | `pixi run -e torch demo-webcam-rerun` | Optional camera/model lane | Webcam/mock perception, model outputs, Rerun logging, record/replay helpers. |
| TWIST2 / MuJoCo | `pixi run -e twist2 demo-twist2-rerun` | Optional simulator lane | Multi-rate simulator/policy/visualization loop. |
| MuJoCo Manipulation | `pixi run -e twist2 python examples/advanced/mujoco_manipulation/app.py` | Source launch file | High-rate physics, slower controller, Rerun state logging. |
| RoboSuite Lift | `pixi run demo-robosuite-mock` | Mock-safe first | Robosuite wrapper and scripted Lift-policy contract. |
| Hierarchical Physics | `pixi run -e twist2 python examples/advanced/hierarchical_physics_demo/app.py --demo both --duration 8` | Source launch file | Explicit clock -> sim -> viz layers plus HTML pipeline visualization. |

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

## MuJoCo Manipulation

Use this lane after the promoted visual checks pass and the TWIST2/MuJoCo environment is available.

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

## RoboSuite Lift

```bash
pixi run demo-robosuite-mock
```

This is the default smoke path: it exercises the Retriever graph contract without requiring robosuite. For a real robosuite `Lift` run, install the optional dependency and run the real mode:

```bash
pixi run python -m pip install -e ".[robosuite]" robosuite
pixi run demo-robosuite-lift
```

What it shows:

- simulator/environment wrapper and policy as separate Flows,
- slow policy updates coupled to faster simulator state with `Latest()`,
- a mock path for CI/docs and a real robosuite path for configured machines.

Source:

- `examples/advanced/robosuite_lift/README.md`
- `examples/advanced/robosuite_lift/app.py`

## Hierarchical Physics

Use this lane for design inspection after the lightweight graph artifact works. The promoted quick graph check remains `pixi run demo-pipeline-html-viz`.

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

## How To Choose

Use mock-safe checks for first-run confidence, Rerun for visual debugging, simulator lanes for rate and environment integration, and source launch files when you are inspecting implementation patterns rather than following the shortest path.
