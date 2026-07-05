# MuJoCo Manipulation Example (Optional Golden Lane)

This is an optional simulator lane for users who already understand the core Retriever runtime and want a robot-control example with mismatched rates. Start with `demo-golden-hub-pack`, `demo-perception-detection-flow`, and `demo-pipeline-html-viz` before using this path.

This example demonstrates unified robotics control using `retriever`:

1. **High-fidelity physics**: `MujocoEnvFlow` runs at 1000 Hz.
2. **Controller**: `ControllerFlow` runs at 50 Hz, typical for RL/inference.
3. **Visualization**: `RerunLoggerFlow` logs to Rerun at 30 Hz.
4. **Explicit coupling**: Retriever sync policies make the physics/controller/logger handoff inspectable instead of hiding it inside callbacks.

## Task: Reach and Chase 🎯

The 2-link arm uses **Inverse Kinematics (Jacobian Transpose Control)** to chase a moving red target sphere.

- **Red Sphere**: Moving Target
- **Blue Trail**: End-Effector Tip
- **Control Law**: $\tau = J^T K_p (x_{target} - x_{tip}) - K_d \dot{q}$

## Usage

```bash
pixi run -e twist2 python examples/advanced/mujoco_manipulation/app.py
```

By default, this example uses the **Dora** backend (`backend="dora"`) to demonstrate high-performance, Rust-based dataflow execution. You can switch back to `backend="multiprocessing"` in `app.py` for pure Python execution.

A **Rerun window** will open showing:
- `camera/render`: The 2-link arm tracking the target.
- `world/target` & `world/tip`: 3D visualization of the tracking performance.
- `state/joint*`: Real-time joint plots.
