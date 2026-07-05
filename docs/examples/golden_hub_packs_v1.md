# Golden Hub Proof v1

<div class="gr-route-pills gr-route-pills-inline">
  <a href="https://openretriever.org/">Retriever home</a>
  <a href="https://openretriever-docs.pages.dev/">Core docs</a>
  <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Visual quickstart</a>
  <a href="https://github.com/openretriever/retriever">Core source</a>
  <a href="/">Golden overview</a>
  <a href="https://github.com/openretriever/golden-retriever">Golden source</a>
  <a href="../llms.txt">Golden agent map</a>
</div>


GoldenRetriever is the maintained reference examples layer for packs loaded through Retriever Hub. The core runtime imports as `retriever`; the public PyPI target is `retriever-core` once the core 0.0.1 release is live. Golden's current manifest-declared Hub surface is the reusable robot-facing type pack and conversion helpers declared in `pyproject.toml`.

## Core boundary

Start with the core visual quickstart if you are new to Retriever: https://openretriever-docs.pages.dev/getting-started/visual-quickstart/. The core runtime provides Flow, Pipeline, registry, IR, and Hub mechanics; Golden provides robot-facing payloads and maintained examples.


For the exact current export list, see [Hub Export Catalog](../hub/export_catalog_v1.md). For source examples that are candidates for future Hub packs, see [Pack Roadmap](../hub/pack_roadmap_v1.md).

## Fast local proof

Run these from the GoldenRetriever repository:

```bash
pixi run demo-golden-hub-pack
pixi run demo-pipeline-html-viz
```

No Golden wheel is required for this source-checkout path. Install the Retriever runtime once, then load Golden's robot-facing payloads through the local Hub manifest. The public reference shape is the same `hub.use("openretriever/golden-retriever:Export")` string used by indexed Hub packs; source checkout is the launch-safe proof until a public index entry is enabled.

The Hub-pack smoke command does four things locally without network access:

1. reads the local `[tool.retriever.module]` manifest,
2. loads representative exports through the runtime Hub loader,
3. checks that robot-facing types such as `WorldState` are visible through the unified registry,
4. round-trips a lightweight `Action` payload through the exported Arrow helpers.

The visualization command validates a small closed-loop graph to IR, prints an ASCII graph, and writes `out/golden_retriever_closed_loop_viz.html`.

Typical output:

```text
Golden pack exports: WorldState, BeliefGraph, Skill, Plan, Trajectory, convert_to_arrow, convert_from_arrow
Registry WorldState: _retriever_hub...WorldState
Constructed WorldState: ['cup']
Constructed Plan skills: ['pick']
Arrow round-trip: Action OK
Retriever Hub reference: hub.use("openretriever/golden-retriever:WorldState")
Graph proof: run `pixi run demo-pipeline-html-viz` to validate and render an IR HTML artifact.
```

## Hub reference

Golden exports use the same reference shape as any other Retriever Hub pack:

```python
from retriever import hub

WorldState = hub.use("openretriever/golden-retriever:WorldState")
Plan = hub.use("openretriever/golden-retriever:Plan")
convert_to_arrow = hub.use("openretriever/golden-retriever:convert_to_arrow")
```

Golden does not need a separate runtime package for this path. Users install the runtime once, then load domain packs through Hub. Source examples are promoted to Hub packs only after they are import-safe, versioned, smoke-tested, and documented.

## What this proves

This page is intentionally narrower than a full robot demo. It proves the
public extension boundary:

- the manifest can be loaded from a standard `src/` layout,
- Golden exports appear as Hub-loaded objects,
- robot-facing payloads register in the runtime registry,
- conversion helpers are exported with the pack,
- graph validation and visualization remain runtime-owned.

For richer examples, continue with perception, memory, language, composition,
and simulation lanes in the Example Guides section.
