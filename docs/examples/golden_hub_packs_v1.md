# Golden Hub Packs v1

GoldenRetriever is the example and type-pack catalog for Retriever. The core
runtime package is `retriever-core` and imports as `retriever`; Golden's reusable
robot-facing payloads are exposed through the Retriever Hub module manifest in
`pyproject.toml`.

## Fast proof from source checkout

Run these from the GoldenRetriever repository:

```bash
pixi run demo-golden-hub-pack
pixi run demo-pipeline-html-viz
```

The Hub-pack smoke command does four things without network access:

1. reads the local `[tool.retriever.module]` manifest,
2. loads representative exports through the runtime Hub loader,
3. checks that applied types such as `WorldState` are visible through the unified registry,
4. round-trips a lightweight `Action` payload through the exported Arrow helpers.

The visualization command validates a small closed-loop graph to IR and writes a
self-contained HTML artifact under `out/`.

Typical output:

```text
Golden Hub exports: WorldState, BeliefGraph, Skill, Plan, Trajectory, convert_to_arrow, convert_from_arrow
Registry WorldState: _retriever_hub...WorldState
Constructed WorldState: ['cup']
Constructed Plan skills: ['pick']
Arrow round-trip: Action OK
Public path after launch: hub.use("openretriever/golden-retriever:WorldState")
Graph proof: run `pixi run demo-pipeline-html-viz` to validate and render an IR HTML artifact.
```

## Public Hub path after launch

After `openretriever/golden-retriever` and the Hub index are public, users should
load Golden exports the same way as any other Hub module:

```python
from retriever import hub

WorldState = hub.use("openretriever/golden-retriever:WorldState")
Plan = hub.use("openretriever/golden-retriever:Plan")
convert_to_arrow = hub.use("openretriever/golden-retriever:convert_to_arrow")
```

Golden does not need a separate first-launch PyPI package for this path. Users
install the runtime once, then load domain packs through Hub.

## What this proves

This page is intentionally narrower than a full robot demo. It proves the
release-critical extension boundary:

- the manifest can be loaded from a standard `src/` layout,
- Golden exports appear as Hub-loaded objects,
- applied payloads register in the runtime registry,
- conversion helpers are exported with the pack,
- graph validation and visualization remain runtime-owned.

For richer examples, continue with perception, memory, language, composition,
and simulation lanes in the Example Guides section.
