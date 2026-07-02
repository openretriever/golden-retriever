# Simulation and Visualization Examples

These examples are the current Golden paths for visual and simulator-oriented demos. They are optional lanes after the concise perception -> memory -> language ladder.

## Quick Map

| Lane | Command | What it shows |
| --- | --- | --- |
| Webcam + Rerun | `pixi run -e torch demo-webcam-rerun` | Webcam/mock perception, model outputs, Rerun logging, record/replay helpers. |
| TWIST2 / MuJoCo | `pixi run -e torch demo-twist2-rerun` | Multi-rate simulator/policy/visualization loop. |
| MuJoCo Manipulation | `pixi run python examples/advanced/mujoco_manipulation/app.py` | High-rate physics, slower controller, Rerun state logging. |
| Hierarchical Physics | `pixi run python examples/advanced/hierarchical_physics_demo/app.py --demo both --duration 8` | Explicit clock -> sim -> viz layers plus HTML pipeline visualization. |
| Web Command Interface | `pixi run python examples/advanced/web_command_interface/app.py` | Local browser-facing command/debug surface. |
| Pipeline HTML Viz | `pixi run python examples/experimental/visualization/visualize_pipeline.py` | Self-contained IR graph export to ASCII and HTML. |

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
pixi run -e torch demo-twist2-rerun
```

For native MuJoCo viewer support on machines configured for it:

```bash
pixi run -e torch demo-twist2
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

```bash
pixi run python examples/advanced/mujoco_manipulation/app.py
```

What it shows:

- high-rate MuJoCo physics,
- slower controller Flow,
- Rerun visualization of camera/render output, target/tip geometry, and joint state plots,
- explicit synchronization between physics, control, and visualization rates.

Source:

- `examples/advanced/mujoco_manipulation/README.md`
- `examples/advanced/mujoco_manipulation/app.py`

## Hierarchical Physics + HTML Pipeline Visualization

```bash
pixi run python examples/advanced/hierarchical_physics_demo/app.py --demo both --duration 8
```

What it shows:

- explicit clock -> simulation -> visualization layers,
- Rerun logging for physics state,
- generated interactive HTML pipeline visualization,
- ASCII pipeline summaries for quick inspection.

Source:

- `examples/advanced/hierarchical_physics_demo/README.md`
- `examples/advanced/hierarchical_physics_demo/app.py`

## Web Command Interface

```bash
pixi run python examples/advanced/web_command_interface/app.py
```

What it shows:

- a small browser-facing control surface,
- command handling around a Retriever-style system boundary,
- a useful pattern for local operator/debug UIs.

Source:

- `examples/advanced/web_command_interface/app.py`
- `examples/advanced/web_command_interface/static/index.html`

## Self-Contained Pipeline HTML Viz

```bash
pixi run python examples/experimental/visualization/visualize_pipeline.py
```

What it shows:

- a small cyclic dummy pipeline,
- IR generation,
- ASCII graph output,
- `closed_loop_viz.html` export.

This stays in `examples/experimental` for now because it is a small visualization utility rather than a polished example family.

## Current Scope

This page only lists examples that exist in the current repository and have concrete launch files. Keep new simulator lanes out of the public front door until they have a runnable command, dependency story, and README.
