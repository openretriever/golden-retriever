# TWIST2 MuJoCo Simulation (Retriever Port)

> **Note**: This example requires the TWIST2 assets. Please clone them into the root of the repo:
> ```bash
> git clone https://github.com/unitreerobotics/TWIST2.git
> ```
> This folder is ignored by default to save space.

Port of the [TWIST2 Humanoid Controller](https://github.com/amazon-far/TWIST2) to **Retriever**, showcasing:

- **Frequency Decoupling**: Physics (1000 Hz) vs Policy (50 Hz)
- **`@gui_flow`**: Native MuJoCo viewer via main-thread execution
- **Dora Backend**: High-performance dataflow with Rust coordination

## Quick Start

```bash
# Native MuJoCo viewer + Rerun (recommended)
pixi run -e torch demo-twist2

# Rerun only (headless, works everywhere)
pixi run -e torch demo-twist2-rerun
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
