# Hierarchical VLA (Vision-Language-Action) Controller

This example demonstrates a **hierarchical control architecture** typical in advanced robotics. It simulates a "slow" Vision-Language Model (VLA) running on a high-latency loop and a "fast" low-level controller running at high frequency.

## Key Concepts

1.  **Mixed-Frequency Execution**:
    -   **Perception (VLA)**: Runs at **~1 Hz**. It processes complex multimodal inputs (text/images) to generate a high-level "Goal Embedding".
    -   **Control**: Runs at **~50 Hz**. It consumes the latest goal embedding and joint states to output high-frequency motor commands.

2.  **Wrappers & Factories**:
    -   Uses `retriever.lib.hf.from_hf` to wrap Hugging Face Transformers.
    -   Demonstrates the **Factory Pattern** (`create_pipeline`) to ensure compatibility with multiprocessing backends like `dora`.

3.  **Dataflow**:
    -   Uses `Rate` adapters to manage the frequency difference.
    -   The Control loop always receives the **latest** available perception output without blocking.

## Architecture

```mermaid
graph TD
    User[Command Generator] -->|Text Command| Perception[Perception VLA]
    Perception -->|Goal Embedding (1 Hz)| Control[Control Policy]
    Robot[Robot Driver] -->|Joint State (50 Hz)| Control
    Control -->|Motor Torques (50 Hz)| Robot
```

## Running the Example

This example requires the `torch` environment.

```bash
# Run with default settings (dora backend, 15s duration)
pixi run -e torch demo-hierarchical-vla

# Run with custom duration
pixi run -e torch demo-hierarchical-vla --duration 30
```

## Implementation Details

-   **`perception.py`**: Wraps a `bert-tiny` model (simulating a VLA) using `from_hf`.
-   **`control.py`**: A PyTorch-based MLP policy that expects a goal vector and joint angles.
-   **`app.py`**: Orchestrates the graph using `connect` and `Latest` strategies.

## Notes on Backends

-   **Dora**: The default and recommended backend. It runs nodes in separate processes, simulating a real distributed robot OS. We use a top-level factory function for the pipeline to ensure it can be pickled and sent to the worker process.

## Design Notes
For the architectural decisions behind this example, see:
- [Future Robotics Examples](../../../../docs/temp_notes/2025-12-21_future_robotics_examples.md#4-hierarchical-vla-vision-language-action-implemented)
