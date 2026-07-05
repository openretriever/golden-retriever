<div class="gr-reference-hero gr-reference-hero-tight">
  <div>
    <p class="gr-eyebrow">Golden examples for Retriever</p>
    <h1>Run applied robot examples after the core runtime quickstart.</h1>
    <p class="gr-lede">Retriever core teaches Flow, Pipeline, clocks, sync, IR, execution, record/replay, and Hub loading. GoldenRetriever is the applied examples layer after that: robot-facing examples, reusable type packs, simulator and visualization lanes, and Hub-pack candidates.</p>
  </div>
  <div class="gr-command-strip">
    <span>First Golden command</span>
    <code>pixi run demo-golden-hub-pack</code>
    <p>Run this from a GoldenRetriever checkout after the core visual quickstart works.</p>
  </div>
</div>

<div class="gr-route-pills gr-route-pills-inline">
  <a href="https://openretriever.org/">Retriever home</a>
  <a href="https://openretriever-docs.pages.dev/">Core docs</a>
  <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Core visual quickstart</a>
  <a href="https://openretriever-docs.pages.dev/tutorials/debug-and-visualize/">Core debugging</a>
  <a href="examples/">Golden examples</a>
  <a href="hub/">Hub packs</a>
  <a href="robotics_typing_standard/">Robot type packs</a>
  <a href="llms.txt">Agent map</a>
</div>

!!! note "Golden is not a second runtime"
    Install and learn the runtime once: `retriever-core`, imported as `retriever`. Golden carries maintained robot-facing examples and packs on top of that same runtime. Runtime mechanics stay in the core Retriever docs; reusable example code and pack candidates live here.

## Recommended Path

<div class="gr-path-grid gr-path-grid-four">
  <a class="gr-path-step" href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">
    <span>00</span>
    <strong>Start in core</strong>
    <p>Run the mock visual graph, then the webcam/Rerun path. This teaches Flow, stepping, graph inspection, and replay without robot dependencies.</p>
    <code>pixi run demo-webcam-detection-mock</code>
  </a>
  <a class="gr-path-step" href="examples/golden_hub_packs_v1/">
    <span>01</span>
    <strong>Prove the Golden boundary</strong>
    <p>Load Golden's manifest-declared robot payloads through Retriever Hub instead of treating Golden as another framework.</p>
    <code>pixi run demo-golden-hub-pack</code>
  </a>
  <a class="gr-path-step" href="examples/">
    <span>02</span>
    <strong>Run applied examples</strong>
    <p>Choose perception, memory, language, composition, simulation, visualization, or robot typing examples with named commands and expected outputs.</p>
    <code>pixi run -e golden-local demo-perception-detection-flow</code>
  </a>
  <a class="gr-path-step" href="hub/module_roadmap_v1/">
    <span>03</span>
    <strong>Promote reusable packs</strong>
    <p>Move examples into Hub-loadable packs only when their import, version, smoke-test, dependency, and docs contracts are stable.</p>
    <code>hub.use("openretriever/golden-retriever:WorldState")</code>
  </a>
</div>

## What Golden Adds

<div class="gr-purpose-grid">
  <div class="gr-purpose-card">
    <span>Applied graphs</span>
    <strong>Robot-facing examples after the core demos</strong>
    <p>Golden turns the core runtime model into maintained examples for perception, memory, language, planning-style payloads, simulator wrappers, and controllers connected as typed Flows.</p>
  </div>
  <div class="gr-purpose-card">
    <span>Reusable packs</span>
    <strong>Robot payload vocabulary</strong>
    <p>WorldState, RobotState, BeliefGraph, Skill, Plan, StructuredPlan, TaskGoal, Trajectory, ExecutionStatus, Action, Command, Status, and Arrow helpers live here as Golden pack exports.</p>
  </div>
  <div class="gr-purpose-card">
    <span>Debug artifacts</span>
    <strong>Evidence before heavier dependencies</strong>
    <p>Start with terminal proofs, mock-safe simulator traces, and self-contained HTML graphs. Then opt into webcam/Rerun, MuJoCo/TWIST2, robosuite, or model-backed lanes when configured.</p>
  </div>
</div>

## Demo Gallery

<div class="gr-demo-grid gr-demo-grid-compact">
  <a class="gr-demo-card gr-demo-card-primary" href="examples/golden_hub_packs_v1/">
    <span>Hub proof</span>
    <strong>Golden loads through Retriever Hub</strong>
    <p>Manifest load, registry visibility, robot-facing payloads, and Arrow helper round-trip.</p>
    <dl>
      <dt>Command</dt><dd><code>pixi run demo-golden-hub-pack</code></dd>
      <dt>Expected result</dt><dd>Terminal export summary; no robot, camera, model, simulator, or network required.</dd>
    </dl>
  </a>
  <a class="gr-demo-card" href="examples/perception_and_memory_v1/">
    <span>Applied ladder</span>
    <strong>Perception -> memory -> language</strong>
    <p>Detections feed belief state, remembered pointing, replay, caption planning, and composed control over deterministic scenes.</p>
    <dl>
      <dt>First command</dt><dd><code>pixi run -e golden-local demo-perception-detection-flow</code></dd>
      <dt>Expected result</dt><dd>Exit-zero flow smokes and readable terminal observations.</dd>
    </dl>
  </a>
  <a class="gr-demo-card" href="examples/simulation_and_visualization_v1/">
    <span>Visualization</span>
    <strong>Graph HTML, webcam/Rerun, and simulator lanes</strong>
    <p>Start with mock-safe graph visualization, then opt into webcam/Rerun, MuJoCo/TWIST2, or robosuite when dependencies are configured.</p>
    <dl>
      <dt>Safe command</dt><dd><code>pixi run demo-pipeline-html-viz</code></dd>
      <dt>Expected result</dt><dd><code>out/golden_retriever_closed_loop_viz.html</code> plus an ASCII graph.</dd>
    </dl>
  </a>
  <a class="gr-demo-card" href="robotics_typing_standard/">
    <span>Type packs</span>
    <strong>Robot payload and dataset contracts</strong>
    <p>Reusable contracts for world state, belief, skills, plans, commands, statuses, trajectories, event streams, and dataset profiles.</p>
    <dl>
      <dt>First command</dt><dd><code>pixi run demo-robotics-typing-catalog</code></dd>
      <dt>Expected result</dt><dd>Typed payload examples reusable across Golden examples and future Hub packs.</dd>
    </dl>
  </a>
</div>

## Example Results To Recognize

<div class="gr-artifact-grid">
  <figure class="gr-figure-card gr-figure-card-wide">
    <img src="assets/robot-agent-graph-ultralight.png" alt="Retriever robot agent graph with observe, belief, planner, monitor, skill, and controller flows" />
    <figcaption><strong>Applied robot graph.</strong> Golden examples make this kind of multi-rate graph concrete: camera, belief, planner, monitor, skill, controller, and environment-like Flows keep their timing boundaries explicit.</figcaption>
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
  <div class="gr-result-card">
    <span>Mock simulator</span>
    <strong>Robot loop without robosuite installed</strong>
    <pre><code>pixi run demo-robosuite-mock
[mock step=00] object_height=... reward=...
[mock step=01] object_height=... reward=...</code></pre>
  </div>
  <div class="gr-result-card">
    <span>Optional visual lane</span>
    <strong>Webcam/Rerun when dependencies exist</strong>
    <pre><code>pixi run -e torch demo-webcam-rerun
# camera or mock frames flow through perception
# Rerun shows frames, detections, and replay artifacts</code></pre>
  </div>
</div>

## Retriever Ecosystem Map

<div class="gr-layer-map">
  <div>
    <span>Landing</span>
    <strong><a href="https://openretriever.org/">openretriever.org</a></strong>
    <p>Compact front door: positioning, first command, and route selection.</p>
  </div>
  <div>
    <span>Core runtime</span>
    <strong><a href="https://openretriever-docs.pages.dev/">Retriever docs</a></strong>
    <p>Flow, Pipeline, clocks, sync, IR, stepping, record/replay, standard types, and Hub mechanics.</p>
  </div>
  <div>
    <span>Golden examples</span>
    <strong>GoldenRetriever</strong>
    <p>Maintained robot-facing examples, type packs, simulator/visualization lanes, notebooks, and Hub-pack candidates.</p>
  </div>
</div>

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

## What Belongs Where

| Surface | Lives in | Why |
| --- | --- | --- |
| Flow, Pipeline, clocks, sync, IR, stepping, replay, execution | Core Retriever | These are runtime semantics and should stay one source of truth. |
| Standard broadly reusable runtime types | Core Retriever | These should be importable as `retriever.types.*` across apps and packs. |
| Applied robot payload packs | GoldenRetriever | These evolve with robot examples and can later become Hub-loadable packs. |
| Camera/model/simulator/robot examples | GoldenRetriever | These need dependency tiers, smoke paths, and expected artifacts. |
| Experimental or hardware-bound lanes | Source-only until promoted | They should not be first-run public entrypoints until dependency and output contracts are clear. |

## Scope Rules

- Keep core runtime API details in `openretriever/retriever`.
- Keep Golden focused on robot-facing examples, tutorials, notebooks, robotics typing, visualization lanes, simulator wrappers, and Hub-pack candidates.
- Keep camera, model-backed, MuJoCo, TWIST2, real robosuite, and hardware-bound lanes optional until their dependency story is explicit.
- Promote a source example into a Hub-loadable pack only after it is import-safe, versioned, smoke-tested, documented, and useful outside this repository.
