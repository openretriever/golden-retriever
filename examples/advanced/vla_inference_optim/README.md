# VLA Temporal Alignment & Inference Optimization

This example demonstrates a robust architecture for integrating **VLA (Vision-Language-Action)** models with high-frequency robot control loops, specifically addressing the challenges of **Inference Latency** and **Jitter**.

## The Problem: "Receding Horizon" with Latency
VLA models (like OpenPI `pi0`) are often heavy, taking **150ms - 300ms** to generate an action chunk.
- **Observation Time ($t_{obs}$)**: When the camera image was captured.
- **Availability Time ($t_{avail}$)**: When the inference finishes.

By the time the robot receives the action chunk, the first few actions in that chunk correspond to time steps that have *already passed*. Executing them blindly causes "laggy" or jerky motion.

## The Solution: "Fast-Forward" Control Strategy
We implement a **Temporal Alignment** logic in the `ActionBuffer` node. Instead of a simple FIFO queue, we treat the action chunk as a time-indexed trajectory.

### 1. Mathematical Logic
To find the correct action index $k$ to execute at the current wall-clock time ($t_{now}$):

1.  **Calculate Elapsed Time**:
    $$ \delta = t_{now} - t_{obs} $$
    *(How much time has passed since the image was taken?)*

2.  **Determine Index**:
    $$ k = \text{round}\left( \frac{\delta}{dt} \right) $$
    *(Which step in the chunk corresponds to "now"?)*

3.  **Execution**:
    - **If $k < 0$**: Impossible (Prediction is from the future).
    - **If $0 \le k < H$**: Valid execution. Execute action $a_k$.
    - **If $k \ge H$**: Chunk Expired. Stop or hold.

## Implementation Variations

We provide two reference implementations to demonstrate different architectural choices:

### 1. Adapter Pattern (`demo_buffer_adapter.py`)
-   **Concept**: Uses a custom **Adapter** (`ActionChunk`) to encapsulate the buffering and interpolation logic.
-   **Graph**: `VLA --(ActionChunk)--> Sink`
-   **Pros**: Cleaner graph topology; composable logic.

### 2. Explicit Flow Pattern (`demo_buffer_flow.py`)
-   **Concept**: Uses a dedicated **Flow Node** (`ActionBuffer`) to manage the state.
-   **Graph**: `VLA --(Latest)--> Buffer --(Latest)--> Sink`
-   **Pros**: Easier to debug state; explicit backpressure handling.

## Verification & Visualization

### Running the Demos
Both demos are configured to run with the `multiprocessing` backend.

**Adapter Version:**
```bash
pixi run python -m examples.advanced.vla_inference_optim.demo_buffer_adapter --backend multiprocessing --duration 2
```

**Explicit Buffer Version:**
```bash
pixi run python -m examples.advanced.vla_inference_optim.demo_buffer_flow --backend multiprocessing --duration 2
```

### Visualizing the Pipeline
Both scripts validate the authored graph to IR for a terminal summary, then call the public `Pipeline.visualize(...)` API to write an **interactive HTML graph**:
-   `vla_pipeline_adapter.html`
-   `vla_pipeline_manual.html`

Open these files in a web browser to inspect node topology, clock domains (Rates), and data connections. The examples no longer call the old `build_ir()` helper directly.

### Expected Output
You should see logs indicating the "Fast-Forward" behavior, where the Sink executes actions with positive indices even immediately after a chunk arrives (compensating for latency).
```
[Sink] Executed 50 steps. Last: 0.583
...
MockVLA Rate: 6.6 Hz (Actual: 151.7ms) ...
```
