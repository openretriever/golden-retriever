# Example Guides

GoldenRetriever is the applied examples surface for Retriever. Use these guides after the core visual quickstart, when you want robot-facing perception, memory, language, composition, simulation, visualization, and reusable type-pack examples.

Design rule: start from standard payloads in Retriever core; put applied robotics/planning payloads in Golden type packs or local example modules only when that boundary is reusable.

## Recommended Path

1. [Perception and Memory](perception_and_memory_v1.md) — detection, belief, replay, and composed control over one small scene.
2. [Language and Grounding](language_and_grounding_v1.md) — captioning, grounded references, and primitive plan text.
3. [Pipeline Composition](pipeline_composition_v1.md) — registry-backed composition and reusable pipeline surfaces.
4. [Golden Hub Packs](golden_hub_packs_v1.md) — source-checkout proof that Golden exports load as Hub packs.
5. [Simulation and Visualization](simulation_and_visualization_v1.md) — webcam/Rerun, MuJoCo/TWIST2, mock-safe robosuite, and HTML pipeline views.

## Maintained Example Families

| Family | What to use it for | First command or guide |
| --- | --- | --- |
| Perception | Detection, segmentation, and pointing over one synthetic scene. | `pixi run -e golden-local demo-perception-detection-flow` |
| Memory | Belief updates, dropout memory, and remembered pointing. | `pixi run -e golden-local demo-memory-belief-flow` |
| Language | Captions, grounded references, and primitive plan text. | `pixi run -e golden-local demo-language-caption-plan` |
| Composition | Registry-backed composition and pipeline-as-Flow surfaces. | `pixi run -e golden-local demo-composable-pipelines` |
| Webcam + Rerun | Webcam/mock perception with live visualization and replay helpers. | [Simulation and Visualization](simulation_and_visualization_v1.md) |
| TWIST2 / MuJoCo | Multi-rate simulator, policy, and visualization loops. | `pixi run -e twist2 demo-twist2-rerun` |
| RoboSuite Lift | Mock-safe robosuite wrapper and optional real robosuite mode. | `pixi run demo-robosuite-mock` |
| Pipeline HTML Viz | Small deterministic IR/HTML graph visualization utility. | `pixi run demo-pipeline-html-viz` |

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
- `examples/experimental/visualization/` — maintained lightweight pipeline HTML utility; keep it narrow until moved under `examples/advanced/`

## Scope Notes

- `examples/advanced/closed_loop_planning/` contains extracted design patterns from an older prototype; it is not a first-run runnable lane.
- Browser-command prototypes and other local operator surfaces should stay source-only until they have a README, named Pixi task, dependency story, and expected output.
- Every promoted page should name its runnable command and expected artifact: terminal output, Rerun viewer, HTML pipeline visualization, or mock-safe simulator trace.
