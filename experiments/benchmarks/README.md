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

First, build the ROS 2 packages using `colcon.

```bash
pixi run -e ros ros-build
```

Then, you can run the standard Python (`rclpy`) and C++ (`rclcpp`) benchmarks.
These use the CycloneDDS middleware to communicate between processes.

```bash
pixi run -e ros benchmark_python
pixi run -e ros benchmark_cpp
```

Additionally, in C++, you can set up node components to communicate between nodes using shared memory (intra-process communication).
For more information, see [this link](https://docs.ros.org/en/kilted/Tutorials/Intermediate/Composition.html).

```bash
pixi run -e ros benchmark_cpp_components
```

In all cases, the benchmarks will indicate when they have completed data collection.
At this point, you should terminate the program yourself with Ctrl+C or equivalent.

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

The lower-level Dora latency helper may also create `benchmark_data.csv` inside
`experiments/benchmarks/dora_benchmark/dora-rs/py-latency/`; that file is a
local generated artifact and is not tracked.

**ROS 2 (Python PubSub):**

First build the code.

```bash
pixi run -e ros benchmark-ros-suite-build
```

Then run manually.

```bash
pixi shell -e ros
source experiments/benchmarks/dora_benchmark/ros2/py_pubsub/install/setup.bash
ros2 run py_pubsub listener & ros2 run py_pubsub talker
```
