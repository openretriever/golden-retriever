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

### 2. Architecture Components

#### A. Mock VLA Node (`mock_vla_node.py`)
Simulates the real-world behavior of a VLA model:
- **Latency Simulation**: Sleeps for 100-200ms (with jitter) to mimic `pi0` inference.
- **Output**: Generates a **32-step trajectory** (Horizon $H=32$, $dt=0.1s$).
- **Timestamping**: Crucially, it tags the output with the **Observation Timestamp ($t_{obs}$)**, not the completion time.

#### B. Action Buffer (`app.py`)
Acts as the "Smart Cache" between VLA and Robot:
- Runs at **50Hz** (high frequency).
- Holds the **latest available** action chunk.
- On every tick, it performs the **Fast-Forward** calculation:
  ```python
  delta_t = now - chunk_start_time
  k = int(round(delta_t / dt))
  action = queue[k] # Skips indices 0..k-1
  ```

## Verification
In our tests (running `python -m examples.advanced.vla_inference_optim.app`), we observe:
- **Inference Latency**: ~140ms
- **Buffer Behavior**: Automatically skips indices 0-3 (past actions) and executes index **4** ($0.4s$ mark), ensuring the robot tracks the trajectory in real-time without delay accumulation.

## Usage
Run the demo:
```bash
pixi run python -m examples.advanced.vla_inference_optim.app
```
