# Closed-Loop Planning Examples

Belief-space planning and VLM-assisted planning prototypes.

## Recommended entrypoints

```bash
# High-level planning with the maintained pixi task
pixi run -e llm demo-highlevel-planning

# Direct script entrypoints for the lighter variants
pixi run python examples/experimental/closed_loop_planning/pipelines/demo_simple_grid.py
pixi run python examples/experimental/closed_loop_planning/pipelines/demo_belief_planning.py
```

## Other pipelines

```bash
pixi run python examples/experimental/closed_loop_planning/pipelines/demo_rise_sim.py
pixi run python examples/experimental/closed_loop_planning/pipelines/demo_spot_real.py
pixi run python examples/experimental/closed_loop_planning/pipelines/demo_manipulation.py
```

## Notes

- `demo_highlevel_planning.py` is the most maintained launch path in this folder.
- `demo_spot_real.py` expects live Spot credentials and hardware-specific dependencies.
- The `notes/` directory contains design notes, not user-facing quickstart material.
