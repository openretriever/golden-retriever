# Benchmarks

The basic benchmarks are directly modified from the [`dora-benchmark`](https://github.com/dora-rs/dora-benchmark) repository.

Here, messages containing variable-size array data are sent between two processes.
The latency between sending and receiving the messages are measured using wall clock time.

## Retriever Benchmarks

```bash
pixi run python experiments/benchmarks/benchmark_retriever.py --backend dora
```

## ROS Benchmarks

```bash
pixi run -e ros build
pixi run -e ros benchmark_python
```

## Plot results

```bash
pixi run python experiments/benchmarks/plot_benchmark_results.py 
```
