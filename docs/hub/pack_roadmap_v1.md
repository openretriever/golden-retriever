# Retriever Hub Pack Roadmap v1

<div class="gr-route-pills gr-route-pills-inline">
  <a href="https://openretriever.org/">Retriever home</a>
  <a href="https://openretriever.org/start/">Start path</a>
  <a href="https://openretriever-docs.pages.dev/">Core docs</a>
  <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Visual quickstart</a>
  <a href="/">Golden overview</a>
  <a href="../llms.txt">Golden agent map</a>
</div>


Golden is the maintained reference examples layer for Retriever Hub pack candidates, but not every example should be pack-loadable immediately. The rule is simple: promote examples into Hub packs only when they are import-safe, versioned, smoke-tested, and useful outside this repo.

## Promotion rule

A Golden source example is ready to become a Retriever Hub pack when it has:

- import-safe top-level code,
- lightweight serializable construction config,
- no camera, robot, simulator, GPU, socket, model-key, or file opening at import time,
- a named Pixi smoke command,
- a documented expected artifact,
- clear dependency level: source-only, optional model, optional simulator, or hardware-bound.

## Current v1 pack-loadable surface

| Pack | Status | Why |
| --- | --- | --- |
| Applied robotics type pack | Live in `pyproject.toml` | Stable, import-safe, useful across examples. |
| Arrow conversion helpers | Live in `pyproject.toml` | Needed for data interchange and smokeable without hardware. |

## Best next Retriever Hub pack candidates

| Candidate | Source lane | First smoke | Promotion notes |
| --- | --- | --- | --- |
| `golden.perception.synthetic_color` | `examples/advanced/perception_examples` | `pixi run -e golden-local demo-perception-detection-flow` | Good first flow pack: deterministic, no hardware, typed payloads. |
| `golden.memory.belief` | `examples/advanced/memory_examples` | `pixi run -e golden-local demo-memory-belief-flow` | Good stateful Flow reference for belief and partial observability. |
| `golden.language.caption_plan` | `examples/advanced/language_examples` | `pixi run -e golden-local demo-language-caption-plan` | Good language payload reference; keep model-backed variants optional. |
| `golden.composition.counter` | `examples/advanced/core_composition` | `pixi run -e golden-local demo-composable-pipelines` | Good pipeline-as-Flow and registry composition reference. |
| `golden.visualization.pipeline_html` | `examples/experimental/visualization` | `pixi run demo-pipeline-html-viz` | Useful as a graph inspection artifact; decide whether it belongs in core or Golden before export. |
| `golden.robosuite.lift_mock` | `examples/advanced/robosuite_lift` | `pixi run demo-robosuite-mock` | Keep mock-first; real robosuite should remain optional. |

## Keep source-only for now

These are useful references but should not be promoted as Hub packs until their dependency and import contracts are tighter:

- unpromoted design-pattern extracts — useful as references, but not first-run runnable lanes.
- model-backed real perception and memory lanes — useful, but require explicit model/key/dependency story.
- simulator-heavy or hardware-bound lanes — keep mock-first and optional.
- browser/operator surfaces — need README, task, expected artifact, and dependency story before promotion.
