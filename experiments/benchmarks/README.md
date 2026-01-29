# Benchmarks

The basic benchmarks are directly modified from the [`dora-benchmark`](https://github.com/dora-rs/dora-benchmark) repository.

Here, messages containing variable-size array data are sent between two processes.
The latency between sending and receiving the messages are measured using wall clock time.

## Retriever Benchmarks

```bash
pixi run python experiments/benchmarks/benchmark_retriever.py --backend dora
pixi run python experiments/benchmarks/benchmark_retriever.py --backend multiprocessing
```

## ROS Benchmarks

```bash
pixi run -e ros build
pixi run -e ros benchmark_python
pixi run -e ros benchmark_cpp
```

## Plot results

```bash
pixi run python experiments/benchmarks/plot_benchmark_results.py 
```

## External Benchmarks (from dora-benchmark)

We also include benchmarks imported from the [dora-benchmark](https://github.com/dora-rs/dora-benchmark) repository.

**Dora (Python Latency):**
```bash
pixi run benchmark-dora-suite
# Results: experiments/benchmarks/results/dora_benchmark_results.csv
```

**ROS 2 (Python PubSub):**
```bash
pixi run -e ros benchmark-ros-suite-build
# Then run manually:
# source experiments/benchmarks/dora_benchmark/ros2/py_pubsub/install/setup.bash
# ros2 run py_pubsub listener & ros2 run py_pubsub talker
```
