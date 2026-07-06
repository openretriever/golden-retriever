# GoldenRetriever

<img src="assets/retriever-illustrative.jpeg" width="120" alt="GoldenRetriever logo">

GoldenRetriever is the applied reference layer for Retriever: robot-facing examples, reusable robot payload references, simulator and visualization lanes, and candidates for future Retriever Hub packs.

Use GoldenRetriever after the core Retriever quickstart. Core Retriever teaches `Flow`, `Pipeline`, clocks, sync, graph rendering, stepping, replay, and Hub loading. GoldenRetriever shows those ideas as runnable robot-facing examples.

## Start Here

```bash
pixi install
pixi run demo-golden-hub-pack
pixi run -e golden-local demo-perception-detection-flow
pixi run demo-pipeline-html-viz
```

Expected results:

- `demo-golden-hub-pack` prints GoldenRetriever payload exports and an Arrow round-trip check.
- `demo-perception-detection-flow` runs a deterministic synthetic perception flow.
- `demo-pipeline-html-viz` writes a self-contained HTML graph artifact.

## Public Docs

- Retriever home: https://openretriever.org/
- Core Retriever docs: https://openretriever-docs.pages.dev/
- GoldenRetriever docs: https://retriever-space.pages.dev/
- First GoldenRetriever proof: https://retriever-space.pages.dev/examples/golden-hub-proof/
- Example catalog: https://retriever-space.pages.dev/examples/
- GoldenRetriever Hub boundary: https://retriever-space.pages.dev/hub/

## Surface Boundary

| Surface | GoldenRetriever owns | Core Retriever owns |
| --- | --- | --- |
| Runtime concepts | Applied examples that use them. | Flow, Pipeline, clocks, sync, IR, replay, execution, and Hub mechanics. |
| Robot examples | Perception, memory, language, composition, simulator, and visualization lanes. | Small runtime/tutorial examples. |
| Reusable Hub packs | Robot payloads, conversion helpers, maturity rules, and Hub pack candidates. | Hub loader and registry mechanics. |

## Repository Map

- `examples/advanced/`: runnable GoldenRetriever example families.
- `src/retriever_typing/`: robot payloads and data/event helpers used by GoldenRetriever examples.
- `docs-site/`: Starlight docs site for GoldenRetriever reference pages.
- `notebooks/`: notebook sources and generated notebook artifacts.

## Common Commands

```bash
pixi run demo-golden-hub-pack
pixi run -e golden-local demo-perception-detection-flow
pixi run -e golden-local demo-memory-belief-flow
pixi run -e golden-local demo-language-caption-plan
pixi run -e golden-local demo-composable-pipelines
pixi run demo-robosuite-mock
pixi run demo-pipeline-html-viz
pixi run demo-robotics-typing-catalog
```

Optional camera, model, simulator, GPU, or network-dependent lanes should come after the mock-safe path works.

Maintainer checks:

```bash
pixi run -e docs docs-build
pixi run public-surface-check
```

## Contributing And License

See `CONTRIBUTING.md` for contribution workflow and `SECURITY.md` for private vulnerability reporting. GoldenRetriever is licensed under Apache License 2.0; see `LICENSE`.
