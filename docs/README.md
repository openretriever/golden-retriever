<div class="gr-reference-hero gr-reference-hero-tight">
  <div>
    <p class="gr-eyebrow">After the Retriever core quickstart</p>
    <h1>Run Retriever on robot-facing examples.</h1>
    <p class="gr-lede">GoldenRetriever is the maintained examples and type-pack layer for Retriever. Learn <code>Flow</code>, clocks, sync policies, debugging, and Hub mechanics in the core docs first; then use this site for perception, memory, language, simulator, visualization, and reusable robotics payload examples.</p>
  </div>
  <div class="gr-command-strip">
    <span>First Golden proof</span>
    <code>pixi run demo-golden-hub-pack</code>
    <p>Loads Golden robot payload exports through Retriever Hub without robot, camera, model, simulator, or network dependencies.</p>
  </div>
</div>

<div class="gr-route-pills gr-route-pills-inline">
  <a href="https://openretriever.org/">Retriever home</a>
  <a href="https://openretriever.org/start/">Builder start path</a>
  <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Core visual quickstart</a>
  <a href="https://openretriever-docs.pages.dev/">Core runtime docs</a>
  <a href="examples/">Golden examples</a>
  <a href="hub/">Hub packs</a>
  <a href="robotics_typing_standard/">Robot type packs</a>
  <a href="https://github.com/openretriever/golden-retriever">Golden source</a>
</div>

!!! note "Golden is not a second runtime"
    Install and learn the runtime once; Python imports use `retriever`. The public PyPI target is `retriever-core`, while release-prep source checkouts use the documented Pixi environments. Golden is the official maintained examples-and-packs surface on top of that runtime. Runtime semantics stay in the core Retriever docs; robot-facing examples, reusable payload packs, notebooks, simulator wrappers, and pack candidates live here.

## Retriever Ecosystem

<div class="gr-layer-map gr-layer-map-compact">
  <div>
    <span>Front door</span>
    <strong><a href="https://openretriever.org/start/">openretriever.org/start</a></strong>
    <p>Use this to choose the shortest path: install, visual quickstart, core docs, Hub, examples, or source.</p>
  </div>
  <div>
    <span>Runtime</span>
    <strong><a href="https://openretriever-docs.pages.dev/">Core Retriever docs</a></strong>
    <p>Owns Flow, Pipeline, clocks, sync policies, stepping, replay, IR, execution, and Hub mechanics.</p>
  </div>
  <div>
    <span>Applied layer</span>
    <strong><a href="examples/">Golden examples</a></strong>
    <p>Owns maintained robot-facing examples, reusable payload packs, simulator/visualization lanes, and pack candidates.</p>
  </div>
</div>

## Recommended Route

<div class="gr-path-grid gr-path-grid-four">
  <a class="gr-path-step" href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">
    <span>00</span>
    <strong>Run core visual quickstart</strong>
    <p>Start with the core webcam/mock visual graph so Flow, stepping, graph inspection, and replay are clear.</p>
    <code>pixi run demo-webcam-detection-mock</code>
  </a>
  <a class="gr-path-step" href="examples/golden_hub_packs_v1/">
    <span>01</span>
    <strong>Prove the Golden boundary</strong>
    <p>Load Golden's manifest-declared robot payloads through Retriever Hub instead of treating Golden as a runtime fork.</p>
    <code>pixi run demo-golden-hub-pack</code>
  </a>
  <a class="gr-path-step" href="examples/">
    <span>02</span>
    <strong>Pick an example lane</strong>
    <p>Choose perception, memory, language, composition, simulation, visualization, or robot typing by command and expected artifact.</p>
    <code>pixi run -e golden-local demo-perception-detection-flow</code>
  </a>
  <a class="gr-path-step" href="hub/pack_roadmap_v1/">
    <span>03</span>
    <strong>Promote reusable packs</strong>
    <p>Move examples into Hub-loadable packs only when imports, versions, smokes, dependency tiers, and docs are stable.</p>
    <code>hub.use("openretriever/golden-retriever:WorldState")</code>
  </a>
</div>

## Command Matrix

Use this table as the Golden first-run checklist. Each row has a command, a dependency level, and the artifact you should recognize.

| Path | Command | Dependency level | Expected artifact | Continue |
| --- | --- | --- | --- | --- |
| Hub proof | `pixi run demo-golden-hub-pack` | Source checkout only | Export list, registry lookup, constructed payloads, Arrow round-trip | [Golden Hub Proof](examples/golden_hub_packs_v1.md) |
| Perception start | `pixi run -e golden-local demo-perception-detection-flow` | Mock-safe Golden env | Deterministic synthetic detections over a small scene | [Perception and Memory](examples/perception_and_memory_v1.md) |
| Memory ladder | `pixi run -e golden-local demo-memory-belief-flow` | Mock-safe Golden env | Belief state update over perception payloads | [Perception and Memory](examples/perception_and_memory_v1.md) |
| Language ladder | `pixi run -e golden-local demo-language-caption-plan` | Mock-safe Golden env | Caption and primitive plan text over typed observations | [Language and Grounding](examples/language_and_grounding_v1.md) |
| Composition | `pixi run -e golden-local demo-composable-pipelines` | Mock-safe Golden env | Registry-backed reusable pipeline composition | [Pipeline Composition](examples/pipeline_composition_v1.md) |
| Graph visualization | `pixi run demo-pipeline-html-viz` | Source checkout only | `out/golden_retriever_closed_loop_viz.html` plus an ASCII graph | [Simulation and Visualization](examples/simulation_and_visualization_v1.md) |
| Robot typing | `pixi run demo-robotics-typing-catalog` | Source checkout only | Typed robot payload examples and registry proof | [Robot Type Packs](robotics_typing_standard/README.md) |
| Mock simulator | `pixi run demo-robosuite-mock` | No real robosuite install | `[mock step=...]` simulator-policy trace lines | [Simulation and Visualization](examples/simulation_and_visualization_v1.md) |
| Optional visual lane | `pixi run -e torch demo-webcam-rerun` | Camera/Rerun/torch env | Frames, detections, and replay-oriented visual traces | [Simulation and Visualization](examples/simulation_and_visualization_v1.md) |

## Example Result Shapes

<div class="gr-artifact-grid gr-artifact-grid-compact">
  <figure class="gr-figure-card gr-figure-card-wide">
    <img src="assets/robot-agent-graph-ultralight.png" alt="Retriever robot agent graph with observe, belief, planner, monitor, skill, and controller flows" />
    <figcaption><strong>Applied robot graph.</strong> Golden examples make multi-rate robot graphs concrete: camera, belief, planner, monitor, skill, controller, and environment-like Flows keep their timing boundaries explicit.</figcaption>
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
    <strong>Self-contained HTML artifact</strong>
    <pre><code>pixi run demo-pipeline-html-viz
# writes out/golden_retriever_closed_loop_viz.html
# prints a compact ASCII graph</code></pre>
  </div>
</div>

## What Belongs Where

| Surface | Lives in | Why |
| --- | --- | --- |
| Product framing and route selection | [openretriever.org](https://openretriever.org/) | Compact public front door: install/run, docs, examples, source. |
| Flow, Pipeline, clocks, sync, IR, stepping, replay, execution | [Retriever core docs](https://openretriever-docs.pages.dev/) | These are runtime semantics and should stay one source of truth. |
| Broadly reusable runtime types | Core Retriever | These should be importable as `retriever.types.*` across apps and packs. |
| Applied robot payload packs | GoldenRetriever | These evolve with robot examples and can later become Hub-loadable packs. |
| Camera/model/simulator/robot examples | GoldenRetriever | These need dependency tiers, smoke paths, and expected artifacts. |
| Experimental or hardware-bound lanes | Source-only until promoted | They should not be first-run public entrypoints until dependency and output contracts are clear. |

## First Commands

Run these from the GoldenRetriever repository after the core quickstart works.

=== "Agent-safe"

    ```bash
    pixi run demo-golden-hub-pack
    pixi run -e golden-local demo-perception-detection-flow
    pixi run demo-robosuite-mock
    pixi run demo-pipeline-html-viz
    pixi run public-surface-check
    ```

=== "Concise ladder"

    ```bash
    pixi run demo-golden-hub-pack
    pixi run -e golden-local demo-perception-detection-flow
    pixi run -e golden-local demo-memory-belief-flow
    pixi run -e golden-local demo-language-caption-plan
    pixi run -e golden-local demo-language-grounded-reference
    pixi run -e golden-local demo-composable-pipelines
    ```

=== "Visualization"

    ```bash
    pixi run demo-pipeline-html-viz
    pixi run demo-robosuite-mock
    pixi run -e torch demo-webcam-rerun
    pixi run -e twist2 demo-twist2-rerun
    ```

=== "Typing"

    ```bash
    pixi run demo-robotics-typing-catalog
    pixi run demo-robotics-typing-contract
    pixi run demo-robotics-typing-boundary
    ```

## Next Pages

<div class="gr-layer-map gr-layer-map-compact">
  <div>
    <span>Examples</span>
    <strong><a href="examples/">Golden Example Catalog</a></strong>
    <p>Command-first guide for perception, memory, language, composition, simulation, visualization, and type-pack lanes.</p>
  </div>
  <div>
    <span>Hub</span>
    <strong><a href="hub/">Retriever Hub Packs</a></strong>
    <p>Pack boundary, current export catalog, and promotion rules for moving source examples into reusable packs.</p>
  </div>
  <div>
    <span>Types</span>
    <strong><a href="robotics_typing_standard/">Robot Type Packs</a></strong>
    <p>World state, belief, skills, plans, commands, statuses, trajectories, event streams, and dataset profiles.</p>
  </div>
</div>
