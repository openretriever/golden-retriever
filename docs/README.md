# GoldenRetriever Hub Module

<div class="gr-hero gr-hero-split">
  <div class="gr-hero-copy">
    <p class="gr-eyebrow">Applied Retriever Hub module</p>
    <h1>Run applied robotics examples as Retriever Hub modules.</h1>
    <p class="gr-lede">GoldenRetriever is the first applied robotics Hub module for the core <code>retriever</code> runtime: reusable robot-facing type packs plus maintained perception, memory, language, composition, visualization, simulator, and robosuite lanes.</p>
    <div class="gr-route-pills">
      <a href="https://openretriever.org/">Landing</a>
      <a href="https://openretriever-docs.pages.dev/">Core docs</a>
      <a href="https://openretriever-docs.pages.dev/getting-started/visual-quickstart/">Core quickstart</a>
      <a href="https://github.com/openretriever/golden-retriever">Source</a>
    </div>
  </div>
  <div class="gr-command-panel">
    <img src="assets/retriever-illustrative.jpeg" alt="GoldenRetriever logo" class="gr-hero-logo" />
    <p class="gr-panel-label">First Golden command</p>
    <pre><code>pixi run demo-golden-hub-pack</code></pre>
    <p class="gr-panel-note">Expected result: the Golden Hub manifest loads, exported types register, and Arrow helpers round-trip locally.</p>
  </div>
</div>

!!! note "Prerequisite"
    If you are new to Retriever, complete the [core visual quickstart](https://openretriever-docs.pages.dev/getting-started/visual-quickstart/) first. The command `pixi run demo-webcam-detection` belongs to `openretriever/retriever`, not this repository. Golden starts after that with `pixi run demo-golden-hub-pack`, then the concise perception and memory ladder.

!!! info "Runtime boundary"
    The core runtime distribution is `retriever-core` and imports as `retriever`. Golden itself is not a second runtime package; reusable applied types are Hub exports, and heavier examples stay in the source docs.

## Recommended Path

<div class="gr-path-grid">
  <a class="gr-path-step" href="examples/golden_hub_packs_v1/">
    <span>01</span>
    <strong>Golden Hub proof</strong>
    <p>Load Golden exports through the runtime Hub loader and verify registry/Arrow behavior.</p>
    <code>pixi run demo-golden-hub-pack</code>
  </a>
  <a class="gr-path-step" href="examples/perception_and_memory_v1/">
    <span>02</span>
    <strong>Perception and memory</strong>
    <p>Detection, replay, belief, memory, and composed control over one small scene.</p>
    <code>pixi run -e golden-local demo-perception-detection-flow</code>
  </a>
  <a class="gr-path-step" href="examples/language_and_grounding_v1/">
    <span>03</span>
    <strong>Language and grounding</strong>
    <p>Captioning, grounded references, and primitive plan-text examples.</p>
    <code>pixi run -e golden-local demo-language-caption-plan</code>
  </a>
  <a class="gr-path-step" href="examples/pipeline_composition_v1/">
    <span>04</span>
    <strong>Composition</strong>
    <p>Registry-backed composition and reusable pipeline surfaces.</p>
    <code>pixi run -e golden-local demo-composable-pipelines</code>
  </a>
  <a class="gr-path-step" href="examples/simulation_and_visualization_v1/">
    <span>05</span>
    <strong>Visualization</strong>
    <p>Rerun, mock-safe robosuite, MuJoCo/TWIST2, and HTML pipeline views.</p>
    <code>pixi run demo-robosuite-mock</code>
  </a>
</div>

## Demo Gallery

Pick one demo by the artifact you want to see. The commands below are the
current promoted Golden surface: mock-safe first, richer simulator/model lanes
only when their optional environment is installed.

<div class="gr-demo-grid">
  <a class="gr-demo-card gr-demo-card-primary" href="examples/golden_hub_packs_v1/">
    <span>Hub proof</span>
    <strong>Load Golden as a Retriever Hub module</strong>
    <p>Proves the extension boundary before any robot demo: manifest load, runtime registry visibility, exported type pack, and Arrow conversion helpers.</p>
    <dl>
      <dt>Command</dt><dd><code>pixi run demo-golden-hub-pack</code></dd>
      <dt>Expected artifact</dt><dd>Terminal summary: exports, registry lookup, constructed payloads, Arrow round-trip.</dd>
      <dt>Dependency level</dt><dd>Source checkout only; no robot, camera, model, or simulator.</dd>
    </dl>
  </a>
  <a class="gr-demo-card" href="examples/perception_and_memory_v1/">
    <span>Concise ladder</span>
    <strong>Perception -> memory over one scene</strong>
    <p>Shows the applied example story without hardware: detections feed belief state and remembered pointing over a small synthetic scene.</p>
    <dl>
      <dt>Command</dt><dd><code>pixi run -e golden-local demo-perception-detection-flow</code></dd>
      <dt>Expected artifact</dt><dd>Small terminal trace of detections or memory updates.</dd>
      <dt>Dependency level</dt><dd>Mock/local path; no camera or robot required.</dd>
    </dl>
  </a>
  <a class="gr-demo-card" href="examples/language_and_grounding_v1/">
    <span>Language</span>
    <strong>Caption, ground, and sketch a plan</strong>
    <p>Connects simple language payloads to grounded references and primitive plan text, after perception/memory are clear.</p>
    <dl>
      <dt>Command</dt><dd><code>pixi run -e golden-local demo-language-caption-plan</code></dd>
      <dt>Expected artifact</dt><dd>Terminal caption/plan text over the shared scene.</dd>
      <dt>Dependency level</dt><dd>Mock/local path; model-backed lanes stay optional.</dd>
    </dl>
  </a>
  <a class="gr-demo-card" href="examples/pipeline_composition_v1/">
    <span>Composition</span>
    <strong>Reuse a pipeline as a module</strong>
    <p>Demonstrates registry-backed composition and reusable pipeline surfaces before heavier robot integrations.</p>
    <dl>
      <dt>Command</dt><dd><code>pixi run -e golden-local demo-composable-pipelines</code></dd>
      <dt>Expected artifact</dt><dd>Terminal proof that composed surfaces run as one reusable unit.</dd>
      <dt>Dependency level</dt><dd>Source checkout only.</dd>
    </dl>
  </a>
  <a class="gr-demo-card" href="examples/simulation_and_visualization_v1/">
    <span>Visual proof</span>
    <strong>Render graphs, Rerun, and simulator lanes</strong>
    <p>Keeps visual demos explicit: mock-safe robosuite, generated HTML pipeline views, and optional MuJoCo/TWIST2/Rerun paths.</p>
    <dl>
      <dt>Command</dt><dd><code>pixi run demo-pipeline-html-viz</code></dd>
      <dt>Expected artifact</dt><dd>Self-contained HTML graph under <code>out/</code>; robosuite mock prints a simulator-policy trace.</dd>
      <dt>Dependency level</dt><dd>HTML and robosuite mock are lightweight; MuJoCo/TWIST2 is optional.</dd>
    </dl>
  </a>
  <a class="gr-demo-card" href="robotics_typing_standard/">
    <span>Interfaces</span>
    <strong>Check robot-facing payload contracts</strong>
    <p>Use this when you need stable type boundaries for reusable examples, datasets, or future Hub modules.</p>
    <dl>
      <dt>Command</dt><dd><code>pixi run demo-robotics-typing-catalog</code></dd>
      <dt>Expected artifact</dt><dd>Terminal catalog of available standard and applied payload types.</dd>
      <dt>Dependency level</dt><dd>Source checkout only.</dd>
    </dl>
  </a>
</div>

## What Belongs Here

<div class="gr-action-grid">
  <a class="gr-action-card" href="examples/">
    <span>Examples</span>
    <strong>Runnable applied lanes</strong>
    <small>Perception, memory, language, composition, simulation, and visualization examples.</small>
  </a>
  <a class="gr-action-card" href="examples/simulation_and_visualization_v1/">
    <span>Visual proof</span>
    <strong>See richer demos</strong>
    <small>Rerun, MuJoCo/TWIST2, robosuite mock, and generated pipeline HTML.</small>
  </a>
  <a class="gr-action-card" href="examples/golden_hub_packs_v1/">
    <span>Hub packs</span>
    <strong>Load reusable payloads</strong>
    <small>Smoke the Golden Hub manifest, registry visibility, and conversion helpers.</small>
  </a>
  <a class="gr-action-card" href="robotics_typing_standard/">
    <span>Typed payloads</span>
    <strong>Use stable interfaces</strong>
    <small>Robot payload contracts and data/event stream profiles for reusable examples.</small>
  </a>
</div>

## First Commands

Run these from the GoldenRetriever repository after the core quickstart works. Expected outputs are small terminal summaries or visual windows depending on the lane; the mock-safe commands should not require robot hardware.

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
    pixi run -e torch demo-webcam-rerun
    pixi run -e twist2 demo-twist2-rerun
    pixi run demo-robosuite-mock
    pixi run demo-pipeline-html-viz
    ```

=== "Typing"

    ```bash
    pixi run demo-robotics-typing-catalog
    pixi run demo-robotics-typing-contract
    pixi run demo-robotics-typing-boundary
    ```

## Public Boundary

Keep the public source-of-truth split explicit:

- Core runtime API details belong in `openretriever/retriever`.
- GoldenRetriever carries examples, tutorials, notebooks, robotics typing, and integration lanes.
- Reusable Golden payloads should be loaded as Retriever Hub packs; Golden is not a separate runtime package.
- Heavy optional examples should stay clearly optional and mock-first where possible.
- Removed or stale experimental prototypes should not be presented as the main path.
