# Functional Wiring and Pipeline Composition

This folder demonstrates lightweight composition patterns for Retriever flows and pipeline fragments.

## Quick Start

```bash
pixi run demo-chaining
pixi run demo-fanin
pixi run demo-fanout
pixi run demo-sync-policies
pixi run demo-perception-belief-control
```

## Examples

- `chaining.py`: sequential flow composition with `then()` / `>>`.
- `fanin.py`: multiple sources into one destination.
- `fanout.py`: one source feeding multiple destinations.
- `combinators.py`: FRP-style `>>` / `&` graph building.
- `sync_policies.py`: alignment and temporal sampling policies.
- `perception_belief_control_pipeline.py`: compose a small perception stage and control stage into one runnable pipeline.

## Recommended pattern

For reusable subgraphs, write small builder functions that return the surfaced flow you want the next stage to consume.

In this folder, `perception_belief_control_pipeline.py` uses exactly that pattern:
- `attach_perception_stage(...)` builds a perception slice and returns the surfaced belief flow.
- `attach_control_stage(...)` consumes that surfaced belief flow and adds the downstream controller slice.

That keeps graph construction explicit without forcing a registry or hub dependency into every example.


## Related stateful examples

To see where this staged-builder pattern plugs into memory-bearing flows, continue with:

- `../state_management/belief_updater_internal.py`
- `../state_management/perception_replay_to_belief.py`

Those examples provide the belief-state surface that can be composed into a downstream control slice.
