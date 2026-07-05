# Golden Hub Proof

GoldenRetriever is the maintained applied examples layer for packs loaded through Retriever Hub. Golden exports robot-facing payloads and conversion helpers from a local manifest today; the same reference shape is what downstream projects use when the pack is indexed.

## What this page proves

This is the smallest Golden check: it proves that a project outside the core runtime can expose reusable robot payloads through Retriever Hub.

```bash
pixi run demo-golden-hub-pack
pixi run demo-pipeline-html-viz
```

The Hub-pack smoke command does four things without network access:

1. reads the `[tool.retriever.module]` manifest,
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

## Hub reference shape

Golden exports use the same reference shape as any Retriever Hub pack:

```python
from retriever import hub

WorldState = hub.use("openretriever/golden-retriever:WorldState")
Plan = hub.use("openretriever/golden-retriever:Plan")
convert_to_arrow = hub.use("openretriever/golden-retriever:convert_to_arrow")
```

Users install the runtime once, then load domain packs through Hub. Golden stays useful because its examples prove that boundary with robot-facing payloads.

## Continue

- For the exact export list, see [Golden Pack Export Catalog](../hub/export_catalog_v1.md).
- For pack maturity rules, see [Pack Maturity Guide](../hub/pack_roadmap_v1.md).
- For richer examples, continue with perception, memory, language, composition, and simulation lanes in the Example Catalog.
