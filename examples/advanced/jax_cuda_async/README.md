# JAX (Flax) Async Examples

This directory contains examples demonstrating how to use **JAX** and **Flax** with the **Retriever** framework, featuring zero-copy data transfer and asynchronous execution.

## Prerequisites

This example runs in the `jax` environment managed by pixi.

```bash
# Run Inference Demo
pixi run -e jax demo-jax-inference

# Run Training Demo
pixi run -e jax demo-jax-train
```

## Files

1.  **`inference.py`**: A simple example showing how to wrap a Flax `nn.Module` using `from_jax` and run inference in a pipeline.
2.  **`train.py`**: An advanced example demonstrating **Split Learning** across two asynchronous processes.
    *   **Source Node**: Holds the first half of the model (Part A) and the Optimizer.
    *   **Compute Node**: Holds the second half (Part B) and computes loss.
    *   **Flow**: Part A -> (Hidden State) -> Part B -> (Loss & Grad) -> (Gradient) -> Part A (Update).

## Zero-Copy Support

If you run these examples with a backend that supports shared memory (like `dora` on Linux/Mac or specific configurations), JAX arrays are transferred with **zero-copy overhead** where possible.

When passing data between nodes:
*   **`JaxIO`** dataclass is used to wrap `jax.numpy.ndarray`.
*   The Retriever framework handles the serialization/deserialization (or shared memory handle passing).

## Key Concepts

### Functional State Management
Unlike PyTorch's `nn.Module`, Flax modules are **stateless**. The parameters are stored separately.

*   **`retriever.lib.jax.from_jax`** handles this wrapper logic for you.
*   It initializes `self.params` on startup using a sample input.
*   In `run()`, it calls `self.module.apply(self.params, input)`.

### Async Training
In `train.py`, we demonstrate how to handle gradients in a functional way:

1.  **Compute Node**: Uses `jax.value_and_grad` to compute gradients with respect to the *input* (hidden state) and *its own parameters* (if any).
2.  **Source Node**: Receives the gradient for the hidden state, and uses it to update its own parameters via `optax`.
