# Golden Example Catalog

<div class="gr-route-pills gr-route-pills-inline">
  <a href="/">Golden overview</a>
  <a href="/examples/">Examples</a>
  <a href="/hub/">Hub packs</a>
  <a href="/robotics_typing_standard/">Robot type packs</a>
</div>

GoldenRetriever is Track G in the Retriever ecosystem: the maintained applied examples layer. Use this catalog after the core visual quickstart when you want concrete robot-facing paths: Hub-pack proof, perception, memory, language, composition, simulation, visualization, and reusable type-pack examples.

Boundary rule: runtime mechanics stay in core Retriever; robot-facing payloads, maintained examples, and reusable pack candidates live here. Export through Retriever Hub only after the pack is import-safe, versioned, smoke-tested, and documented.

## Recommended Path

<div class="gr-path-grid gr-path-grid-five">
  <a class="gr-path-step" href="golden_hub_packs_v1/">
    <span>01</span>
    <strong>Golden Hub proof</strong>
    <p>Proves Golden extends Retriever through a manifest and type-pack exports.</p>
    <code>pixi run demo-golden-hub-pack</code>
  </a>
  <a class="gr-path-step" href="perception_and_memory_v1/">
    <span>02</span>
    <strong>Perception</strong>
    <p>Runs deterministic detection, segmentation, and pointing over one small scene.</p>
    <code>pixi run -e golden-local demo-perception-detection-flow</code>
  </a>
  <a class="gr-path-step" href="perception_and_memory_v1/">
    <span>03</span>
    <strong>Memory</strong>
    <p>Adds belief updates and remembered pointing over the same typed payloads.</p>
    <code>pixi run -e golden-local demo-memory-belief-flow</code>
  </a>
  <a class="gr-path-step" href="language_and_grounding_v1/">
    <span>04</span>
    <strong>Language</strong>
    <p>Turns typed observations into captions, grounded references, and primitive plans.</p>
    <code>pixi run -e golden-local demo-language-caption-plan</code>
  </a>
  <a class="gr-path-step" href="pipeline_composition_v1/">
    <span>05</span>
    <strong>Composition</strong>
    <p>Shows registry-backed composition and reusable pipeline surfaces.</p>
    <code>pixi run -e golden-local demo-composable-pipelines</code>
  </a>
</div>

## First Verified Commands

Use this sequence when a new user needs high-signal proof without optional hardware or model dependencies:

| Command | Expected result | Why it comes first |
| --- | --- | --- |
| `pixi run demo-golden-hub-pack` | Prints Golden pack exports, registry lookup, constructed payloads, and Arrow round-trip. | Shows Golden extending Retriever through a manifest instead of becoming a second runtime package. |
| `pixi run -e golden-local demo-perception-detection-flow` | Steps a deterministic synthetic perception graph to completion. | Proves the concise example ladder runs before heavier integrations. |
| `pixi run demo-robosuite-mock` | Prints `[mock step=...]` simulator-policy trace lines. | Proves environment-as-Flow and policy-as-Flow without robosuite installed. |
| `pixi run demo-pipeline-html-viz` | Prints an ASCII graph and writes `out/golden_retriever_closed_loop_viz.html`. | Proves IR validation and graph inspection on a closed-loop example. |

Optional camera, model-backed, MuJoCo, TWIST2, and real robosuite lanes should be selected only after the mock-safe path is green.

## Example Results

<div class="gr-artifact-grid gr-artifact-grid-compact">
  <figure class="gr-figure-card gr-figure-card-wide">
    <img src="../assets/robot-agent-graph-ultralight.png" alt="Retriever robot agent graph with observe, belief, planner, monitor, skill, and controller flows" />
    <figcaption><strong>Closed-loop graph shape.</strong> The applied examples combine observation, belief, planning, monitoring, skill execution, control, and environment-like state while keeping timing and sync explicit.</figcaption>
  </figure>
  <div class="gr-result-card">
    <span>Hub proof</span>
    <strong>Terminal export summary</strong>
    <pre><code>Golden pack exports: WorldState, BeliefGraph, Skill, Plan, Trajectory, ...
Registry WorldState: ...WorldState
Arrow round-trip: Action OK</code></pre>
  </div>
  <div class="gr-result-card">
    <span>Graph proof</span>
    <strong>HTML graph artifact</strong>
    <pre><code>pixi run demo-pipeline-html-viz
# out/golden_retriever_closed_loop_viz.html
# ASCII graph printed in terminal</code></pre>
  </div>
</div>

## Maintained Example Families

| Family | What to use it for | Core concept it demonstrates | First command or guide |
| --- | --- | --- | --- |
| Hub proof | Load Golden exports through Retriever Hub and the unified runtime registry. | Retriever Hub pack + robot-facing type pack | `pixi run demo-golden-hub-pack` |
| Perception | Detection, segmentation, and pointing over one synthetic scene. | Flow I/O + typed perception payloads | `pixi run -e golden-local demo-perception-detection-flow` |
| Memory | Belief updates, dropout memory, and remembered pointing. | Local Flow state + replayable inputs | `pixi run -e golden-local demo-memory-belief-flow` |
| Language | Captions, grounded references, and primitive plan text. | Typed language payloads across stages | `pixi run -e golden-local demo-language-caption-plan` |
| Composition | Registry-backed composition and pipeline-as-Flow surfaces. | Reusable graphs and Hub-style boundaries | `pixi run -e golden-local demo-composable-pipelines` |
| Webcam + Rerun | Webcam/mock perception with live visualization and replay helpers. | Debugging and visualization | [Simulation and Visualization](simulation_and_visualization_v1.md) |
| TWIST2 / MuJoCo | Multi-rate simulator, policy, and visualization loops. | Clocks and sync across simulator/policy rates | `pixi run -e twist2 demo-twist2-rerun` |
| RoboSuite Lift | Mock-safe robosuite wrapper and optional real robosuite mode. | Environment-as-Flow and policy-as-Flow | `pixi run demo-robosuite-mock` |
| Pipeline HTML Viz | Maintained promoted IR/HTML graph visualization utility. | IR validation and graph inspection | `pixi run demo-pipeline-html-viz` |
| Robot type packs | Robot payload contracts and dataset/event stream profiles. | Reusable robotics types | `pixi run demo-robotics-typing-catalog` |

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
