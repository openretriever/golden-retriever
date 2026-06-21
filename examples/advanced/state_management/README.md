# State Management Reference

Primary learning path now lives in `../memory_examples/README.md`. This folder is the more detailed reference for reset behavior, replay-driven state updates, and older stateful variants.

## Quick Start

```bash
pixi run demo-stateful-reset
pixi run demo-belief-updater-internal
pixi run demo-perception-replay-to-belief
```

## Primary path

- `stateful_flow_reset.py`: the smallest internal-state example, showing exactly what `pipe.reset()` resets.
- `belief_updater_internal.py`: internal-memory belief update inside one flow.
- `perception_replay_to_belief.py`: replayed perception outputs feeding a stateful belief update stage.

## Advanced variations

- `belief_updater_explicit.py`: explicit-state belief update where state is passed through the graph.
- `stateful_replanning.py`: internal planner memory that emits plan updates only when obstacle events change.
- `localization_eff.py`: effectful localization with deterministic sensor inputs.
- `object_tracking_eff.py`: effectful multi-object tracking with deterministic detections.

These files are reference surfaces for state and memory structure. Stable cross-example/process payloads should still come from the shared basic type surfaces.

## Related guides

- Main memory path: `../memory_examples/README.md`
- Upstream perception path: `../perception_examples/README.md`
- Debug/replay follow-on: `../perception_debug/README.md`
- Downstream composition: `../functional_wiring/README.md`
