# PyTorch Async Split-Learning Example

A split-learning example showing how Retriever can move tensor payloads between stages with minimal serialization overhead.

## Run the application

```bash
pixi run -e torch demo-pytorch-async
```

## Run the benchmark

```bash
pixi run -e torch python examples/advanced/pytorch_cuda_async/benchmark.py --output benchmark_results.csv --plot benchmark_plot.png
```
