# State Management Examples

These examples focus on memory, reset behavior, and deterministic stepper workflows.

## Quick Start

```bash
pixi run demo-stateful-reset
pixi run demo-belief-updater-internal
pixi run demo-perception-replay-to-belief
pixi run demo-belief-updater-explicit
```

## Examples

- `stateful_flow_reset.py`: the smallest internal-state example, showing exactly what `pipe.reset()` resets.
- `belief_updater_internal.py`: internal-memory belief update inside one flow.
- `belief_updater_explicit.py`: explicit-state belief update where state is passed through the graph.
- `perception_replay_to_belief.py`: replayed perception outputs feeding a stateful belief update stage.
- `localization_eff.py`: effectful localization with deterministic sensor inputs.
- `object_tracking_eff.py`: effectful multi-object tracking with deterministic detections.

## Related guides

- `../perception_debug/README.md` for the perception-side record/replay workflow.
- `../functional_wiring/README.md` for the composition pattern used after belief state becomes stable.
