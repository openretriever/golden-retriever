# Retriever Hub Packs

<div class="gr-route-pills gr-route-pills-inline">
  <a href="/">Golden overview</a>
  <a href="/examples/">Examples</a>
  <a href="/hub/">Hub packs</a>
  <a href="/robotics_typing_standard/">Robot type packs</a>
</div>


GoldenRetriever is the maintained reference examples layer and proving ground for Retriever Hub packs. It shows how a repository outside the runtime exposes reusable robot-facing payloads through Retriever Hub while keeping heavier examples as source-level reference lanes until they are ready for stable pack contracts.

## Mental model

```text
Retriever core runtime
  import package `retriever`; Flow, Pipeline, clocks, sync policies, IR, runtime, Hub loader

GoldenRetriever
  robot-facing payloads, example pipelines, simulator wrappers,
  visualization demos, notebooks, and promotion candidates for future Retriever Hub packs
```

The current v1 Golden Hub contract is intentionally small and stable: robot-facing types plus conversion helpers. That gives downstream examples a common vocabulary without turning Golden into a second runtime package or forcing users to copy example code.

## Core Hub vs Golden packs

| Surface | Responsibility |
| --- | --- |
| Core Retriever docs | Hub reference syntax, loader behavior, publishing rules, cache expectations, and generic module boundaries. |
| GoldenRetriever | Applied robot payload packs, example-backed export catalog, and promotion candidates that prove the Hub boundary on real robot-facing examples. |

Use the core docs to learn the protocol. Use Golden to see what a maintained, example-backed pack should look like before depending on it from a robot project.


## Pack lifecycle

```text
source example -> promoted demo -> Hub-loadable pack -> downstream dependency
```

A source example can be useful before it is a pack. A Hub-loadable pack needs a tighter contract: no heavyweight import side effects, versioned exports, a named smoke command, documented expected output, and a clear dependency level.

## Load a Golden pack through Retriever Hub

The Retriever Hub reference shape is:

```python
from retriever import hub

WorldState = hub.use("openretriever/golden-retriever:WorldState")
Plan = hub.use("openretriever/golden-retriever:Plan")
convert_to_arrow = hub.use("openretriever/golden-retriever:convert_to_arrow")
```

During local development and when working from source, `pixi run demo-golden-hub-pack` exercises the same manifest without network access.

## What is pack-loadable today

See [Export Catalog](export_catalog_v1.md) for the exact exports declared in `pyproject.toml`.

## Promotion candidates

See [Pack Maturity Guide](pack_roadmap_v1.md). The short version: promote source examples into Hub packs only when they are import-safe, have serializable construction config, and have a documented smoke command.


## What stays source-only

Golden source examples are useful before they become Hub packs. Keep camera, model-backed, simulator-heavy, hardware-bound, or operator-facing lanes as source-checkout examples until they have a documented dependency level, an import-safe construction path, and a smoke command.
