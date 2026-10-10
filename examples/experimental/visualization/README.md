# Experimental Pipeline Visualization

Small self-contained visualization utilities for inspecting Retriever IR graphs without robot hardware, models, or simulator dependencies.

## Run

From the GoldenRetriever repository root:

```bash
pixi run demo-pipeline-html-viz
```

Expected output:

- an ASCII graph printed to the terminal,
- `out/golden_retriever_closed_loop_viz.html`, a self-contained interactive HTML graph.

The demo builds a tiny cyclic pipeline (`env -> perception -> planner -> executor -> env`) and exports its IR. It is intentionally synthetic: use it to inspect graph rendering, cycles, clocks, ports, and sync policies before running heavier examples.

## Animated Export

```bash
pixi run demo-pipeline-animated-viz
```

Writes `out/golden_retriever_animated_graph.html`, preserving original Flow
class names and instance IDs. Its motion illustrates configured clocks, not
recorded execution. The graph is exported without running the pipeline.

See [Animated pipeline graph](README_animated_graph.md) for the export API,
layout controls, offline behavior, and animation limits.

## Scope

This stays under `examples/experimental/` because it is a utility lane, not a polished robotics example family. Keep it dependency-light and deterministic. If it grows into a promoted example, move it into `examples/advanced/` and add it to the main example guide.
