# GoldenRetriever

System-level integrations, advanced examples, and research prototypes built on top of the core `retriever` runtime.

## Setup

### Default packaged core

```bash
pixi install
```

Use this for the portable Golden examples that rely on the bundled `retriever_dist` wheel.

### Local editable core (for debugging against `retriever-mirror`)

```bash
pixi install -e golden-local
pixi run -e golden-local python -c "import retriever; print(retriever.__file__)"
```

Use this when you want Golden to resolve `retriever` from the sibling `../retriever-mirror` checkout while keeping the public/default environment unchanged.

## Recommended Launch Points

```bash
pixi run demo-robotics-typing-catalog
pixi run demo-synthetic-color-stepper
pixi run demo-perception-record
pixi run demo-perception-replay
pixi run -e golden-local demo-detection-window-stats
pixi run demo-stateful-reset
pixi run demo-belief-updater-internal
pixi run demo-belief-updater-explicit
pixi run -e golden-local demo-stateful-replanning
pixi run demo-perception-replay-to-belief
pixi run demo-perception-belief-control
pixi run -e golden-local demo-composable-pipelines
pixi run demo-multi-agent-communication
pixi run -e tamp demo-tamp-tabletop
```

The three `golden-local` launch points above are the ones in this list that require the local editable-core environment.

## Repository Layout

- `src/golden_retriever`: system integrations, robot drivers, and domain-specific runtime glue.
- `src/retriever_typing`: typed robotics and event/data helpers used by several advanced demos.
- `examples/advanced`: runnable advanced demos with concrete launch points. Start with `examples/advanced/README.md`.
- `notebooks`: git-friendly notebook sources and generated notebook artifacts. Start with `notebooks/README.md`.
- `examples/experimental`: heavier prototypes that are still valuable, but less polished.
- `docs/robotics_typing_standard`: typed payload and data-profile notes for this repo.

## Example Families

- `examples/advanced/perception_debug`: synthetic perception, windowed stats, MCAP recording, and replay.
- `examples/advanced/state_management`: internal state, reset behavior, and memory-oriented flows.
- `examples/advanced/functional_wiring`: flow composition, fan-in/fan-out, staged builders, and sync policies. These examples keep payloads simple and structural instead of inventing a new IO wrapper per stage.
- `examples/advanced/core_composition`: registry-backed pipeline composition surfaces that are easiest to explore from the local editable-core env. The intended pattern is stable shared payloads plus structural rewiring, not pipeline-specific envelope classes.

For the end-to-end perception -> memory -> composition walkthrough, see `docs/examples/perception_memory_composition_v1.md`. For the newer registry-backed composition surfaces, continue with `docs/examples/core_composition_surfaces_v1.md`.
- `examples/advanced/multi_agent_communication`: a compact coordination/composition example.
- `examples/advanced/tamp_tabletop_pick_place`: tabletop TAMP with a PyBullet-backed simulator. This example now reuses shared symbolic core types from `retriever-tamp` instead of maintaining a separate local planning type universe.

## Typed Payload Demos

```bash
pixi run demo-robotics-typing-catalog
pixi run demo-robotics-typing-contract
pixi run demo-robotics-typing-boundary
```

For runnable type/data examples, start with `examples/advanced/robotics_typing_standard/README.md`.

## Notebook workflow

```bash
pixi run notebook-to-ipynb-demo
pixi run notebook-to-ipynb-hub
```

These tasks regenerate the git-friendly Jupytext notebooks. `retriever_demo` is a mechanics notebook for the packaged Golden environment. `hub_demo` is the Hub-first notebook and is meant to be *run* from the local editable-core env:

```bash
pixi install -e golden-local
pixi run -e golden-local demo-hub-notebook-source
```

The Hub notebook reads published module refs from environment variables instead of hardcoding any private or organization-specific module names.
