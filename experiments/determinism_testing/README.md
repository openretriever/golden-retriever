# Functional Determinism Experiments

Experiments comparing Retriever's deterministic execution against pub/sub-style nondeterministic execution using a bouncing ball hybrid system.

## Quick Start

### Benchmark (Semantic Demonstration)

Compares Retriever event-time semantics vs pub/sub arrival-time semantics:

```bash
# Quick demo (20 runs, ~10 seconds)
pixi run -e torch determinism-benchmark-quick

# Full benchmark (100 runs, ~30 seconds)
pixi run -e torch determinism-benchmark

# Generate plots from results
pixi run -e torch determinism-plot
```

### Pipeline (Actual Pipeline.run())

Uses real Retriever Flows and Pipeline.run() with gradient computation inside Flows:

```bash
# In-process backend (recommended, 10 runs)
pixi run -e torch determinism-pipeline

# Dora backend (distributed execution, 5 runs)
pixi run -e torch determinism-pipeline-dora
```

**Note**: The in-process backend is recommended for the pipeline version as it allows
direct access to results. The dora backend demonstrates that gradient computation
works within distributed Flows, but collecting results from worker processes requires
additional infrastructure (e.g., shared memory, result queues).

## Custom Parameters

### Benchmark

```bash
python experiments/determinism_testing/bouncing_ball_benchmark.py --help

# Examples:
python experiments/determinism_testing/bouncing_ball_benchmark.py \
    -K 50 \
    --theta 4.0 \
    --jitter-prob 0.3 \
    --show-traces
```

### Pipeline

```bash
python experiments/determinism_testing/bouncing_ball_pipeline.py --help

# Examples:
python experiments/determinism_testing/bouncing_ball_pipeline.py \
    -K 10 \
    --backend in-process \
    --horizon 100
```

## Gradient Verification

Verify that PyTorch gradients are correct and understand why pub/sub gives wrong gradients:

```bash
pixi run -e torch python experiments/determinism_testing/verify_gradient.py
```

**Key Finding**: Pub/sub computes gradients of corrupted trajectories:

- True gradient (Retriever): -0.005496
- Pub/Sub mean gradient: +0.307997 (wrong sign, 57× larger!)

This is why arrival-time semantics breaks gradient-based learning.

## Output

Results saved to `experiments/determinism_testing/results/`:

- `determinism_results.csv` - Per-run data
- `gradient_histogram.png` - Main figure
- `gradient_histogram.pdf` - Publication quality

## Implementation

- `bouncing_ball_benchmark.py` - Semantic comparison (event-time vs arrival-time)
- `bouncing_ball_pipeline.py` - Actual Pipeline.run() with Flow-based gradient computation
- `plot_determinism_results.py` - Visualization script
- `verify_gradient.py` - Gradient verification via finite differences

## Documentation

See detailed analysis and results in:
- `/Users/zlf/ProjectsRemote/2024-Retriever/RetrieverNotes/experiments/determinism/EXPERIMENT_REPORT.md`
- `/Users/zlf/ProjectsRemote/2024-Retriever/RetrieverNotes/experiments/determinism/SUMMARY.md`
