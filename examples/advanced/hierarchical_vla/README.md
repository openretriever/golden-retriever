# Hierarchical VLA Controller

A mixed-frequency perception/control demo: a slow VLA-style perception stage drives a faster low-level controller.

## Quick Start

```bash
# Recommended local path
pixi run -e torch demo-hierarchical-vla

# Direct command with explicit backend choice
pixi run -e torch python examples/advanced/hierarchical_vla/app.py --backend multiprocessing --duration 15
pixi run -e torch python examples/advanced/hierarchical_vla/app.py --backend dora --duration 15
```

## What it demonstrates

- slow perception and fast control in one graph
- `Latest()` sampling between mismatched rates
- wrapping model-style components behind Retriever flows

`multiprocessing` is the easiest local backend. Use `dora` when you specifically want the distributed/runtime split.
