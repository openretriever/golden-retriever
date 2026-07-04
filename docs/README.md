# GoldenRetriever Examples

<div class="gr-hero gr-hero-split">
  <div class="gr-hero-copy">
    <p class="gr-eyebrow">Applied examples for Retriever</p>
    <h1>Run robot-facing examples after the core runtime quickstart.</h1>
    <p class="gr-lede">GoldenRetriever is the companion examples and Hub pack surface for the core <code>retriever</code> runtime: concise perception, memory, language, composition, visualization, simulator lanes, and reusable robot-facing type packs.</p>
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
    <pre><code>pixi install -e golden-local
pixi run -e golden-local demo-perception-detection-flow</code></pre>
    <p class="gr-panel-note">Expected result: a small terminal demo over the shared synthetic scene. No robot hardware required.</p>
  </div>
</div>

!!! note "Prerequisite"
    If you are new to Retriever, complete the [core visual quickstart](https://openretriever-docs.pages.dev/getting-started/visual-quickstart/) first. The command `pixi run demo-webcam-detection` belongs to `openretriever/retriever`, not this repository. Golden starts after that with `pixi run -e golden-local demo-perception-detection-flow`.

!!! info "Runtime boundary"
    The core runtime distribution is `retriever-core` and imports as `retriever`. Golden itself is not a second runtime package; reusable applied types and examples are exposed through Retriever Hub and source docs. Use `RETRIEVER_CORE_SRC` only when validating unreleased runtime changes locally.

## Recommended Path

<div class="gr-path-grid">
  <a class="gr-path-step" href="examples/perception_and_memory_v1/">
    <span>01</span>
    <strong>Perception and memory</strong>
    <p>Detection, replay, belief, memory, and composed control over one small scene.</p>
    <code>pixi run -e golden-local demo-perception-detection-flow</code>
  </a>
  <a class="gr-path-step" href="examples/language_and_grounding_v1/">
    <span>02</span>
    <strong>Language and grounding</strong>
    <p>Captioning, grounded references, and primitive plan-text examples.</p>
    <code>pixi run -e golden-local demo-language-caption-plan</code>
  </a>
  <a class="gr-path-step" href="examples/pipeline_composition_v1/">
    <span>03</span>
    <strong>Composition</strong>
    <p>Registry-backed composition and reusable pipeline surfaces.</p>
    <code>pixi run -e golden-local demo-composable-pipelines</code>
  </a>
  <a class="gr-path-step" href="examples/simulation_and_visualization_v1/">
    <span>04</span>
    <strong>Visualization</strong>
    <p>Rerun, mock-safe robosuite, MuJoCo/TWIST2, and HTML pipeline views.</p>
    <code>pixi run demo-robosuite-mock</code>
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
    pixi run -e golden-local demo-perception-detection-flow
    pixi run -e golden-local demo-memory-belief-flow
    pixi run -e golden-local demo-language-caption-plan
    pixi run -e golden-local demo-language-grounded-reference
    pixi run -e golden-local demo-composable-pipelines
    pixi run demo-golden-hub-pack
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
