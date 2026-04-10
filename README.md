# GoldenRetriever

System-level integrations, advanced examples, and research prototypes built on top of the core `retriever` runtime.

## Setup

```bash
pixi install
```

## Recommended Launch Points

```bash
pixi run demo-robotics-typing-catalog
pixi run demo-synthetic-color-stepper
pixi run demo-perception-record
pixi run demo-perception-replay-to-belief
pixi run demo-perception-belief-control
pixi run demo-multi-agent-communication
pixi run -e tamp demo-tamp-tabletop
```

## Repository Layout

- `src/golden_retriever`: system integrations, robot drivers, and domain-specific runtime glue.
- `src/retriever_typing`: typed robotics and event/data helpers used by several advanced demos.
- `examples/advanced`: runnable advanced demos with concrete launch points.
- `examples/experimental`: heavier prototypes that are still valuable, but less polished.
- `docs/robotics_typing_standard`: typed payload and data-profile notes for this repo.

## Example Families

- `examples/advanced/perception_debug`: synthetic perception, MCAP recording, and replay.
- `examples/advanced/state_management`: internal state, reset behavior, and memory-oriented flows.
- `examples/advanced/functional_wiring`: flow composition, fan-in/fan-out, staged builders, and sync policies.
- `examples/advanced/multi_agent_communication`: a compact coordination/composition example.
- `examples/advanced/tamp_tabletop_pick_place`: tabletop TAMP with a PyBullet-backed simulator.

## Typed Payload Demos

```bash
pixi run demo-robotics-typing-catalog
pixi run demo-robotics-typing-contract
pixi run demo-robotics-typing-boundary
```

For runnable type/data examples, start with `examples/advanced/robotics_typing_standard/README.md`.
