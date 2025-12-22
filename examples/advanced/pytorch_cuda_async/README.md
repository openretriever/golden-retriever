# PyTorch Async Split-Learning Example

This example demonstrates how to implement distributed split-learning using `retriever` with zero-copy tensor transfer optimizations for both CUDA and CPU.

## Overview

The application splits a neural network into two parts hosted on separate logical nodes:
1.  **Source Node (Part A)**: Generates input data, runs the first layers, and sends activations. It also performs the backward pass.
2.  **Compute Node (Part B)**: Receives activations, runs the remaining layers, computes loss, and sends gradients back.

## Key Features

- **Zero-Copy Transfer**:
    - **CUDA**: Uses `dora.cuda` to transfer GPU memory handles directly (IPC), avoiding CPU round-trips.
    - **CPU**: Uses `numpy` conversion which `retriever`'s Dora backend transmits as zero-copy PyArrow arrays.
- **Asynchronous Execution**: Nodes run in their own frequency/availability, decoupled by the flow runtime.

## Files

- `app.py`: The complete application containing the Pipeline definition, Flow logic, and Zero-Copy utilities.
- `benchmark.py`: A script to measure the raw throughput of the zero-copy wrapper and plot results.
- `ZERO_COPY_GUIDE.md`: A detailed guide on how the zero-copy mechanism works under the hood.

## Running

**1. Run the Application:**
```bash
pixi run -e torch demo-pytorch-async
# Or manually:
# pixi run -e torch python examples/advanced/pytorch_cuda_async/app.py --backend dora
```

**2. Run Benchmarks:**
```bash
pixi run python examples/advanced/pytorch_cuda_async/benchmark.py --output benchmark_results.csv --plot benchmark_plot.png
```
