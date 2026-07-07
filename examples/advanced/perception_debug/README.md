# Perception Debug Follow-Ups

Primary learning path now lives in `../perception_examples/README.md`. This folder is the narrower debug/replay follow-on path once the basic detection, segmentation, and pointing surfaces are clear.

## Quick Start

```bash
pixi run demo-synthetic-color-stepper
pixi run demo-perception-record
pixi run demo-perception-replay
```

## Ordered progression

### 1. Step deterministic perception

```bash
pixi run demo-synthetic-color-stepper
# or
pixi run python examples/advanced/perception_debug/synthetic_color_stepper.py --steps 12 --dt 0.1
```

Use this first when you want one deterministic image source, one detector, and one stepper surface.

### 2. Record and replay the same perception session

```bash
pixi run demo-perception-record
pixi run demo-perception-replay
```

This is the main reason this folder exists: stable artifacts for debugging and downstream belief updates.

## Additional example

### Windowed detection stats

```bash
pixi run -e golden-retriever demo-detection-window-stats
```

This adds a temporal aggregation stage on top of the same synthetic detector. Keep it as a follow-on after the concise perception ladder.

## Where to go next

- Main perception path: `../perception_examples/README.md`
- Stateful follow-on after replay: `../memory_examples/README.md`
- Older state-focused reference: `../state_management/README.md`
