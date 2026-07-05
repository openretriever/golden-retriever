# Golden Example Catalog

<div class="gr-route-pills gr-route-pills-inline">
  <a href="https://openretriever.org/">Retriever home</a>
  <a href="https://openretriever.org/start/">Start path</a>
  <a href="https://openretriever-docs.pages.dev/">Core docs</a>
  <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Visual quickstart</a>
  <a href="https://github.com/openretriever/retriever">Core source</a>
  <a href="/">Golden overview</a>
  <a href="https://github.com/openretriever/golden-retriever">Golden source</a>
  <a href="../llms.txt">Golden agent map</a>
</div>


GoldenRetriever is the maintained reference examples layer for Retriever. Use these guides after the core visual quickstart when you want concrete robot-facing paths: Hub-pack proof, perception, memory, language, composition, simulation, visualization, and reusable type-pack examples.

Boundary rule: runtime mechanics stay in core Retriever; robot-facing payloads, maintained examples, and reusable pack candidates live here. Export through Retriever Hub only after the pack is import-safe, versioned, smoke-tested, and documented.

If this is your first Retriever run, start with the core visual quickstart first. Golden assumes you already understand why a Flow has local state, why clocks are explicit, and why graph/replay artifacts matter.

## Where This Page Fits

- Learn runtime mechanics in the [core docs](https://openretriever-docs.pages.dev/).
- Use this section to pick runnable Golden example lanes.
- Use [Hub Reference](../hub/README.md) when you need the pack/export boundary.
- Use [Robot Type Packs](../robotics_typing_standard/README.md) when you need payload and dataset contracts.

## Recommended Path

1. [Golden Hub Proof](golden_hub_packs_v1.md) — local proof that Golden exports load through the Retriever Hub manifest.
2. [Hub Export Catalog](../hub/export_catalog_v1.md) — exact current downloadable surface declared in `pyproject.toml`.
3. [Perception and Memory](perception_and_memory_v1.md) — detection, belief, replay, and composed control over one small scene.
4. [Language and Grounding](language_and_grounding_v1.md) — captioning, grounded references, and primitive plan text.
5. [Pipeline Composition](pipeline_composition_v1.md) — registry-backed composition and reusable pipeline surfaces.
6. [Simulation and Visualization](simulation_and_visualization_v1.md) — webcam/Rerun, MuJoCo/TWIST2, mock-safe robosuite, and HTML pipeline views.

## Agent-Safe Public Surface

Use this sequence when an agent, CI job, or new user needs high-signal proof without optional hardware or model dependencies:

| Command | Expected result | Why it comes first |
| --- | --- | --- |
| `pixi run demo-golden-hub-pack` | Prints Golden pack exports, registry lookup, constructed payloads, and Arrow round-trip. | Proves Golden extends Retriever through a manifest instead of becoming a second runtime package. |
| `pixi run -e golden-local demo-perception-detection-flow` | Steps a deterministic synthetic perception graph to completion. | Proves the concise example ladder runs before heavier integrations. |
| `pixi run demo-robosuite-mock` | Prints `[mock step=...]` simulator-policy trace lines. | Proves environment-as-Flow and policy-as-Flow without robosuite installed. |
| `pixi run demo-pipeline-html-viz` | Prints an ASCII graph and writes `out/golden_retriever_closed_loop_viz.html`. | Proves IR validation and graph inspection on a closed-loop example. |
| `pixi run public-surface-check` | Prints PASS/FAIL lines for promoted paths, tasks, docs markers, and short runtime smokes. | Prevents the public surface from drifting while heavier lanes evolve. |

Optional camera, model-backed, MuJoCo, TWIST2, and real robosuite lanes should be selected only after the mock-safe path is green.

## Maintained Example Families

| Family | What to use it for | Core concept it demonstrates | First command or guide |
| --- | --- | --- | --- |
| Hub proof | Load Golden exports through Retriever Hub and the unified runtime registry. | Retriever Hub pack + robot-facing type pack | `pixi run demo-golden-hub-pack` |
| Hub export catalog | Inspect current exports and next pack candidates. | Retriever Hub boundary | [Retriever Hub Packs](../hub/README.md) |
| Perception | Detection, segmentation, and pointing over one synthetic scene. | Flow I/O + typed perception payloads | `pixi run -e golden-local demo-perception-detection-flow` |
| Memory | Belief updates, dropout memory, and remembered pointing. | Local Flow state + replayable inputs | `pixi run -e golden-local demo-memory-belief-flow` |
| Language | Captions, grounded references, and primitive plan text. | Typed language payloads across stages | `pixi run -e golden-local demo-language-caption-plan` |
| Composition | Registry-backed composition and pipeline-as-Flow surfaces. | Reusable graphs and Hub-style boundaries | `pixi run -e golden-local demo-composable-pipelines` |
| Webcam + Rerun | Webcam/mock perception with live visualization and replay helpers. | Debugging and visualization | [Simulation and Visualization](simulation_and_visualization_v1.md) |
| TWIST2 / MuJoCo | Multi-rate simulator, policy, and visualization loops. | Clocks and sync across simulator/policy rates | `pixi run -e twist2 demo-twist2-rerun` |
| RoboSuite Lift | Mock-safe robosuite wrapper and optional real robosuite mode. | Environment-as-Flow and policy-as-Flow | `pixi run demo-robosuite-mock` |
| Pipeline HTML Viz | Maintained promoted IR/HTML graph visualization utility. | IR validation and graph inspection | `pixi run demo-pipeline-html-viz` |


## Maturity Levels

| Level | Meaning | User expectation |
| --- | --- | --- |
| Hub-loadable pack | Declared in `pyproject.toml` and loaded by Retriever Hub. | Safe to import and reuse as a pack boundary. |
| Promoted demo | Has a named Pixi task, docs page, expected output, and smoke coverage. | Safe for first-run documentation and CI checks. |
| Source reference | Useful implementation pattern, but not yet a public launch point. | Read the source after the promoted path works. |
| Optional integration | Requires camera, model, simulator, GPU, robot, or external service. | Use only when the dependency story is explicit. |

## Source Folders

Use source folders when you need implementation details after choosing a guide:

- `examples/advanced/perception_examples/`
- `examples/advanced/memory_examples/`
- `examples/advanced/language_examples/`
- `examples/advanced/perception_debug/`
- `examples/advanced/state_management/`
- `examples/advanced/core_composition/`
- `examples/advanced/webcam_rerun/`
- `examples/advanced/twist2_simulation/`
- `examples/advanced/mujoco_manipulation/`
- `examples/advanced/robosuite_lift/`
- `examples/advanced/hierarchical_physics_demo/`
- Pipeline HTML visualization is implemented in `examples/experimental/visualization/`, but the public surface is the promoted `demo-pipeline-html-viz` task and this docs page.

## Scope Notes

- Design-pattern extracts, browser-command prototypes, and other local operator surfaces should stay source-only until they have a README, named Pixi task, dependency story, and expected output.
- Every promoted page should name its runnable command and expected artifact: terminal output, Rerun viewer, HTML pipeline visualization, or mock-safe simulator trace.
