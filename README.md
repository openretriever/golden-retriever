# GoldenRetriever

Advanced examples, system integrations, and research prototypes built on top of the core `retriever` runtime.

## Setup

### Default packaged core

```bash
pixi install
```

Use this for portable Golden examples. Until the public `retriever` package is published, the default environment resolves the core runtime from the temporary `debug-retriever` PyPI package.

### Golden demo environments

```bash
pixi install -e golden-local
pixi run -e golden-local python -c "import retriever; print(retriever.__file__)"
```

The `golden-local` and `golden-perception` names are retained for existing launch commands. For now they resolve the temporary `debug-retriever` runtime package.

## Recommended Launch Points

```bash
pixi run demo-robotics-typing-catalog
pixi run -e golden-local demo-perception-detection-flow
pixi run -e golden-local demo-perception-segmentation-flow
pixi run -e golden-local demo-perception-pointing-flow
pixi run -e golden-local demo-memory-belief-flow
pixi run -e golden-local demo-memory-dropout-flow
pixi run -e golden-local demo-memory-pointing-flow
pixi run -e golden-perception demo-gemini-detection-flow
pixi run -e golden-perception demo-belief-from-real-detections
pixi run demo-perception-record
pixi run demo-perception-replay
pixi run -e golden-local demo-detection-window-stats
pixi run demo-perception-replay-to-belief
pixi run demo-perception-belief-control
pixi run -e golden-local demo-composable-pipelines
pixi run demo-multi-agent-communication
```

The `golden-local` and `golden-perception` launch points use the same temporary packaged runtime plus the Golden example features needed by those commands.

## Repository Layout

- `src/retriever_typing`: typed robotics and event/data helpers used by several advanced demos.
- `examples/advanced`: runnable advanced demos with concrete launch points. Start with `examples/advanced/README.md`.
- `notebooks`: git-friendly notebook sources and generated notebook artifacts. Start with `notebooks/README.md`.
- `examples/experimental`: heavier prototypes that are still valuable, but less polished.
- `docs/robotics_typing_standard`: typed payload and data-profile notes for this repo.
- `docs`: public topic-based docs. Start with `docs/README.md`; `mkdocs.yml` provides a hostable site map.

## Example Families

- `examples/advanced/perception_examples`: concise detection, segmentation, and pointing flows over one shared synthetic scene.
- `examples/advanced/memory_examples`: concise belief and remembered-pointing flows over the same perception payloads.
- `examples/advanced/perception_debug`: synthetic perception, windowed stats, MCAP recording, and replay.
- `examples/advanced/state_management`: older internal state, reset behavior, and memory-oriented flows.
- `examples/advanced/real_memory`: optional explicit real/mock memory flows built on the same detection and belief payloads.
- `examples/advanced/functional_wiring`: flow composition, fan-in/fan-out, staged builders, and sync policies. These examples keep payloads simple and structural instead of inventing a new IO wrapper per stage.
- `examples/advanced/core_composition`: registry-backed pipeline composition surfaces. The intended pattern is stable shared payloads plus structural rewiring, not pipeline-specific envelope classes.
- `examples/advanced/multi_agent_communication`: a compact coordination/composition example.

For the end-to-end perception -> memory -> composition walkthrough, see `docs/examples/perception_and_memory_v1.md`. For the newer registry-backed composition surfaces, continue with `docs/examples/pipeline_composition_v1.md`.

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

These tasks regenerate the git-friendly Jupytext notebooks. `retriever_demo` is a mechanics notebook for the packaged Golden environment. `hub_demo` is the Hub-first notebook and is meant to be run from the Golden demo environment:

```bash
pixi install -e golden-local
pixi run -e golden-local demo-hub-notebook-source
```

The Hub notebook reads published module refs from environment variables instead of hardcoding any private or organization-specific module names.


## Documentation Site

The docs are structured so they can be hosted as a companion website for the core Retriever docs.

```bash
pixi run -e docs docs-build
pixi run -e docs docs-serve
```

Keep Golden docs example-first: concise perception, memory, language, composition, and robotics typing guides belong here; core runtime API details belong in the main `retriever` repo.

## Relationship To Core Retriever

GoldenRetriever is the companion examples and system-integration repository. The core runtime, API reference, and backend implementation live in the main `retriever` repository:

- Runtime repository: `https://github.com/openretriever/retriever`
- Runtime docs: `https://openretriever.github.io/retriever/`
- Golden docs target: `https://openretriever.github.io/golden-retriever/`

Until the public `retriever` package is published, Golden uses the temporary `debug-retriever` runtime package for portable demo environments.

## Contributing And License

See `CONTRIBUTING.md` for the contribution workflow and `SECURITY.md` for private vulnerability reporting. GoldenRetriever is licensed under the Apache License 2.0; see `LICENSE`.
