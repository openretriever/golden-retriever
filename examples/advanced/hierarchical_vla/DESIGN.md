# Design Notes: Hierarchical VLA

## Concept

This example demonstrates a "Skill Switching" or "Hierarchical" architecture where high-level reasoning and low-level control operate at drastically different frequencies.

### Architecture

1.  **Perception (Slow Loop)**:
    *   **Rate**: ~1-5 Hz.
    *   **Component**: `PerceptionFlow` (wraps a Transformer/VLA).
    *   **Role**: Processes heavy multimodal inputs (images, text commands) to produce a "Goal" or "Latent Plan".
    *   **Simulation**: We use `bert-tiny` or similar lightweight transformers to simulate the compute load of a VLM.

2.  **Control (Fast Loop)**:
    *   **Rate**: ~50-100 Hz.
    *   **Component**: `ControlFlow` (MLP Policy).
    *   **Role**: Real-time stabilization and tracking.
    *   **Input**: Consumes the *latest available* Goal from Perception and the *latest* robot state.

### Decoupling Strategy

We use `retriever`'s `Latest` strategy on the connection from Perception to Control.
*   **Control** does *not* wait for a new Goal to step. It reuses the last received Goal.
*   **Perception** runs at its own pace, publishing Goals when ready.

This ensures the robot never stutters due to vision latency.
