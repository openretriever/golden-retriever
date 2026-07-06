<div align="center">

<a href="https://retriever-space.pages.dev/"><img width="360" height="auto" src="assets/retriever-illustrative.jpeg" alt="GoldenRetriever logo"></a>

<br><br>

<a href="https://retriever-space.pages.dev/"><img src="assets/goldenretriever-wordmark.svg" alt="GoldenRetriever" width="540"></a>

### Applied Robot Examples and Hub Packs for Retriever

<p>Robot-facing examples, reusable payload types, simulator and visualization lanes, and Retriever Hub packs — the applied layer on top of the core <code>retriever</code> runtime.</p>

<p>
  <a href="https://retriever-space.pages.dev/"><img alt="Docs" src="https://img.shields.io/badge/Docs-open-b45309?style=for-the-badge"></a>
  <a href="https://retriever-space.pages.dev/examples/"><img alt="Applied examples" src="https://img.shields.io/badge/Applied_Examples-catalog-f97316?style=for-the-badge"></a>
  <a href="https://retriever-space.pages.dev/hub/"><img alt="Hub packs" src="https://img.shields.io/badge/Hub-packs-9333ea?style=for-the-badge"></a>
  <br>
  <a href="https://github.com/openretriever/golden-retriever"><img alt="Source" src="https://img.shields.io/badge/Source-GitHub-111827?style=for-the-badge&logo=github"></a>
  <a href="https://github.com/openretriever/retriever"><img alt="Core runtime" src="https://img.shields.io/badge/Core-retriever-0f766e?style=for-the-badge&logo=github"></a>
  <a href="https://openretriever.org/"><img alt="Website" src="https://img.shields.io/badge/Website-openretriever.org-111827?style=for-the-badge"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-3b82f6?style=for-the-badge"></a>
</p>

</div>

---

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
