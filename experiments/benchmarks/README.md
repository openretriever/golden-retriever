# Benchmarks

## Retriever Benchmarks

```bash
pixi run python experiments/benchmarks/benchmark_retriever.py --backend dora
```

## ROS Benchmarks

```bash
pixi run -e ros build
pixi run -e ros benchmarks_python
```

## Plot results

```bash
pixi run python experiments/benchmarks/plot_benchmark_results.py 
```
