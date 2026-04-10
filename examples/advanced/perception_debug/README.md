# Perception Debug Examples

Lightweight perception examples focused on deterministic stepping, breakpoints, and record/replay workflows.

## Quick Start

```bash
pixi run demo-synthetic-color-stepper
pixi run demo-perception-record
pixi run demo-perception-replay
```

## Examples

### 1. Synthetic Color Stepper

Minimal perception loop with a synthetic image source, a simple color detector, and an in-process stepper.

```bash
pixi run demo-synthetic-color-stepper
# or
pixi run python examples/advanced/perception_debug/synthetic_color_stepper.py --steps 12 --dt 0.1
```

### 2. Record + Replay Perception

Record a short synthetic perception session to MCAP, then replay it through the same detector without re-running the source.

```bash
pixi run demo-perception-record
pixi run demo-perception-replay
```

## Why this folder exists

The heavier webcam/model demos are useful once hardware and model downloads are available. These examples give you the same debugging surface without requiring cameras, large models, or live APIs.


## Next step: replay into memory

After `demo-perception-record` / `demo-perception-replay`, continue with:

```bash
pixi run demo-perception-replay-to-belief
```

That gives you the same perception artifact flowing into a stateful belief update stage instead of stopping at detector outputs.
