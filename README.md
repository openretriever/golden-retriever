# GoldenRetriever

Companion examples, system integrations, and research prototypes built on top of the core `retriever` runtime.

**Links:** [Core runtime](https://github.com/openretriever/retriever) · [Core docs](https://openretriever.github.io/retriever/) · [Golden docs](https://openretriever.github.io/golden-retriever/) · [Website (coming soon)](#) · [Paper / arXiv (coming soon)](#) · [Discord (coming soon)](#)

GoldenRetriever is the examples and integration repository. Keep core runtime API details in [`openretriever/retriever`](https://github.com/openretriever/retriever); use this repo for runnable perception, memory, language, composition, robotics typing, notebook, and system-integration examples.

> Compatibility note: until the public `pyretriever` runtime package is published, the portable Golden environments resolve the temporary packaged runtime configured in `pixi.toml`. The Python import remains `retriever`.

## Setup

```bash
pixi install
```

For the main Golden example environment:

```bash
pixi install -e golden-local
pixi run -e golden-local python -c "import retriever; print(retriever.__file__)"
```

`golden-local` and `golden-perception` are retained as stable launch environments for existing demo commands.

## Recommended Launch Points

Start with the short perception -> memory -> language -> composition ladder:

```bash
pixi run -e golden-local demo-perception-detection-flow
pixi run -e golden-local demo-memory-belief-flow
pixi run -e golden-local demo-language-caption-plan
pixi run -e golden-local demo-language-grounded-reference
pixi run -e golden-local demo-composable-pipelines
```

Then use topic-specific commands as needed:

```bash
pixi run demo-robotics-typing-catalog
pixi run -e golden-local demo-perception-segmentation-flow
pixi run -e golden-local demo-perception-pointing-flow
pixi run -e golden-local demo-memory-dropout-flow
pixi run -e golden-local demo-memory-pointing-flow
pixi run -e golden-perception demo-gemini-detection-flow
pixi run -e golden-perception demo-belief-from-real-detections
pixi run demo-perception-record
pixi run demo-perception-replay
pixi run -e golden-local demo-detection-window-stats
pixi run demo-perception-replay-to-belief
pixi run demo-perception-belief-control
pixi run demo-multi-agent-communication
```

Some heavier optional Pixi environments still resolve demo-only dependencies from Git repositories. Those dependencies are intentionally kept in `pixi.toml`, not the Python package metadata, so default docs and concise examples remain installable without cloning extra research stacks.

## Repository Layout

- `examples/advanced`: runnable advanced demos with concrete launch points. Start with `examples/advanced/README.md`.
- `docs/examples`: public example-guide articles. Start with `docs/examples/README.md`.
- `src/retriever_typing`: typed robotics and event/data helpers used by advanced demos.
- `docs/robotics_typing_standard`: typed payload and data-profile notes for this repo.
- `notebooks`: git-friendly notebook sources and generated notebook artifacts. Start with `notebooks/README.md`.
- `examples/experimental`: heavier prototypes that are still valuable, but less polished.
- `docs`: public topic-based docs; `mkdocs.yml` provides a hostable site map.

## Example Families

- `examples/advanced/perception_examples`: concise detection, segmentation, and pointing flows over one shared synthetic scene.
- `examples/advanced/memory_examples`: concise belief and remembered-pointing flows over the same perception payloads.
- `examples/advanced/language_examples`: caption, grounding, and primitive plan-text examples.
- `examples/advanced/perception_debug`: synthetic perception, windowed stats, MCAP recording, and replay.
- `examples/advanced/real_perception`: optional model-backed perception lanes with mock-first defaults.
- `examples/advanced/real_memory`: optional real/mock memory flows built on detection and belief payloads.
- `examples/advanced/state_management`: older internal state, reset behavior, and memory-oriented flows.
- `examples/advanced/functional_wiring`: flow composition, fan-in/fan-out, staged builders, and sync policies.
- `examples/advanced/core_composition`: registry-backed pipeline composition surfaces.
- `examples/advanced/multi_agent_communication`: a compact coordination/composition example.
- `examples/advanced/closed_loop_planning`: extracted design patterns from the removed experimental prototype.

For the end-to-end perception -> memory -> composition walkthrough, see `docs/examples/perception_and_memory_v1.md`. For registry-backed composition surfaces, continue with `docs/examples/pipeline_composition_v1.md`.

## Typed Payload Demos

```bash
pixi run demo-robotics-typing-catalog
pixi run demo-robotics-typing-contract
pixi run demo-robotics-typing-boundary
```

For runnable type/data examples, start with `examples/advanced/robotics_typing_standard/README.md`.

## Notebook Workflow

```bash
pixi run notebook-to-ipynb-demo
pixi run notebook-to-ipynb-hub
```

These tasks regenerate the git-friendly Jupytext notebooks. `retriever_demo` is a mechanics notebook for the packaged Golden environment. `hub_demo` is the Hub-first notebook and is meant to be run from the Golden demo environment:

```bash
pixi install -e golden-local
pixi run -e golden-local demo-hub-notebook-source
```

The Hub notebook reads published module refs from environment variables instead of hardcoding private or organization-specific module names.

## Documentation Site

The docs are structured so they can be hosted as a companion website for the core Retriever docs.

```bash
pixi run -e docs docs-build
pixi run -e docs docs-serve
```

Keep Golden docs example-first: concise perception, memory, language, composition, and robotics typing guides belong here; core runtime API details belong in the main `retriever` repo.

## Validation

Use these checks before opening a release branch or publishing docs:

```bash
pixi run -e docs docs-build
pixi run -e golden-local test
pixi run -e golden-local demo-perception-detection-flow
pixi run build
```

See `RELEASE.md` for the launch, docs deployment, and package boundary checklist.

## Relationship To Core Retriever

- PyPI distribution: `pyretriever`
- Python import: `retriever`
- Runtime repository: `https://github.com/openretriever/retriever`
- Runtime docs: `https://openretriever.github.io/retriever/`
- Golden repository: `https://github.com/openretriever/golden-retriever`
- Golden docs target: `https://openretriever.github.io/golden-retriever/`

## Contributing And License

See `CONTRIBUTING.md` for the contribution workflow and `SECURITY.md` for private vulnerability reporting. GoldenRetriever is licensed under the Apache License 2.0; see `LICENSE`.
