# TWIST2 MuJoCo Simulation (Optional Golden Lane)

This is an optional simulator/reference lane for users who already ran the core Retriever visual quickstart and the Golden Hub proof. It is not the first Golden command because it depends on MuJoCo/TWIST2 assets and visualization packages.

> **Note**: This example auto-downloads TWIST2 assets into a local cache folder
> (`assets/twist2`) on first run. You can still point to your own asset paths with
> `--xml/--policy/--motion`.

Port of the [TWIST2 Humanoid Controller](https://github.com/amazon-far/TWIST2) to **Retriever**, showcasing:

- **Frequency decoupling**: physics at 1000 Hz, policy at 50 Hz, visualization at 30 Hz.
- **GUI isolation**: `@gui_flow` keeps native MuJoCo viewer work on the main thread.
- **Backend portability**: the same Retriever graph can be inspected locally and run through Dora-backed execution when the dependency stack is available.

## Quick Start

```bash
# Rerun only; safest first simulator smoke.
pixi run -e twist2 demo-twist2-rerun

# Native MuJoCo viewer + Rerun.
pixi run -e twist2 demo-twist2

# Optional: disable auto-download and use only local files.
pixi run -e twist2 python examples/advanced/twist2_simulation/app.py --no-auto-download
```

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│MotionPlayerFlow │────▶│ Twist2PolicyFlow │────▶│  Twist2EnvFlow  │
│    (50 Hz)      │     │     (50 Hz)      │     │   (1000 Hz)     │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                               ┌────────────────────┐
                                               │  @gui_flow         │
                                               │  Twist2VisFlow     │
                                               │  (Main Thread)     │
                                               │  - MuJoCo Viewer   │
                                               │  - Rerun Logging   │
                                               └────────────────────┘
```

### Flows

| Flow | Rate | Description |
|------|------|-------------|
| `Twist2EnvFlow` | 1000 Hz | MuJoCo physics + PD control |
| `Twist2PolicyFlow` | 50 Hz | ONNX policy inference |
| `MotionPlayerFlow` | 50 Hz | Motion reference streaming |
| `Twist2VisFlow` | 30 Hz | **`@gui_flow`** - Native viewer + Rerun |

### `@gui_flow` Feature

The `Twist2VisFlow` uses `@gui_flow` decorator to run in the main thread:

```python
from retriever.flow import gui_flow

@gui_flow
class Twist2VisFlow(Flow[VisInput, None]):
    def init(self):
        # Native GL rendering works because we're in main thread
        self.viewer = mujoco.viewer.launch_passive(...)
```

This enables native MuJoCo visualization without external ZMQ bridges.

## Requirements

- `mujoco` (with `mjpython` on macOS)
- `onnxruntime`
- `rerun-sdk`
