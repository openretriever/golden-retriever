<div align="center">

<a href="https://golden.retriever.build/"><img width="200" height="auto" src="assets/retriever-illustrative.jpeg" alt="GoldenRetriever logo"></a>

<br>

<a href="https://golden.retriever.build/"><img src="assets/goldenretriever-wordmark.svg" alt="GoldenRetriever" width="360"></a>

### Applied Robot Examples and Hub Packs for Retriever

<p>Robot-facing examples, reusable payload types, simulator and visualization lanes, and Retriever Hub packs — the applied layer on top of the core <code>retriever</code> runtime.</p>

<p>
  <a href="https://golden.retriever.build/"><img alt="Docs" src="https://img.shields.io/badge/Docs-open-b45309?style=for-the-badge"></a>
  <a href="https://golden.retriever.build/examples/"><img alt="Applied examples" src="https://img.shields.io/badge/Applied_Examples-catalog-f97316?style=for-the-badge"></a>
  <a href="https://golden.retriever.build/hub/"><img alt="Hub packs" src="https://img.shields.io/badge/Hub-packs-9333ea?style=for-the-badge"></a>
  <br>
  <a href="https://github.com/openretriever/golden-retriever"><img alt="Source" src="https://img.shields.io/badge/Source-GitHub-111827?style=for-the-badge&logo=github"></a>
  <a href="https://github.com/openretriever/retriever"><img alt="Core runtime" src="https://img.shields.io/badge/Core-retriever-0f766e?style=for-the-badge&logo=github"></a>
  <a href="https://openretriever.org/"><img alt="Website" src="https://img.shields.io/badge/Website-openretriever.org-111827?style=for-the-badge"></a>
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache_2.0-3b82f6?style=for-the-badge"></a>
</p>

</div>

---

GoldenRetriever is the applied reference layer for Retriever: robot-facing examples, reusable robot payload types, simulator and visualization lanes, and candidates for future Retriever Hub packs.

Use GoldenRetriever after the core Retriever quickstart. Core Retriever teaches `Flow`, `Pipeline`, clocks, sync, graph rendering, stepping, replay, and Hub loading. GoldenRetriever shows those ideas as runnable robot-facing examples.

## Start Here

GoldenRetriever runs on the core `retriever` runtime and CLI. Install the runtime once, then run the examples from a source checkout:

```bash
pip install retriever-core                 # provides the `retriever` command
git clone https://github.com/openretriever/golden-retriever
cd golden-retriever
retriever install                          # sets up the example environment
retriever run demo-golden-hub-pack         # payload exports + Arrow round-trip
retriever run demo-pipeline-html-viz       # writes a self-contained HTML graph
```

Expected results:

- `demo-golden-hub-pack` prints GoldenRetriever payload exports and an Arrow round-trip check.
- `demo-pipeline-html-viz` writes a self-contained HTML graph artifact.

### Examples that need extra model dependencies

Perception, memory, and language examples pull in heavier libraries, so they live in the `golden-retriever` Pixi environment. Name it explicitly:

```bash
pixi run -e golden-retriever demo-perception-detection-flow
```

GoldenRetriever uses [Pixi](https://pixi.sh) as its environment manager, exactly like the core runtime. `retriever run` wraps Pixi for the default examples; the heavier lanes name the `golden-retriever` environment directly.

## Public Docs

- Retriever home: https://openretriever.org/
- Core Retriever docs: https://retriever.build/
- GoldenRetriever docs: https://golden.retriever.build/
- First GoldenRetriever proof: https://golden.retriever.build/examples/golden-hub-proof/
- Example catalog: https://golden.retriever.build/examples/
- GoldenRetriever Hub boundary: https://golden.retriever.build/hub/

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
retriever run demo-golden-hub-pack
pixi run -e golden-retriever demo-perception-detection-flow
pixi run -e golden-retriever demo-memory-belief-flow
pixi run -e golden-retriever demo-language-caption-plan
pixi run -e golden-retriever demo-composable-pipelines
retriever run demo-robosuite-mock
retriever run demo-pipeline-html-viz
retriever run demo-robotics-typing-catalog
```

Optional camera, model, simulator, GPU, or network-dependent lanes should come after the mock-safe path works.

Maintainer checks:

```bash
pixi run -e docs docs-build
pixi run public-surface-check
```

## Clone and Stay in Sync

```bash
git clone https://github.com/openretriever/golden-retriever
cd golden-retriever
git pull   # normal pulls fast-forward
```

`main` is canonical and only fast-forwards. If a pre-release clone won't pull
(history was consolidated once before release), reset to the published line:

```bash
git fetch origin
git reset --hard origin/main   # discards local commits on the old history
```

## Contributing And License

See `CONTRIBUTING.md` for contribution workflow and `SECURITY.md` for private vulnerability reporting. GoldenRetriever is licensed under Apache License 2.0; see `LICENSE`.
