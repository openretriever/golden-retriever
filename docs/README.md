# GoldenRetriever Examples

<div class="gr-hero">
  <img src="assets/retriever-illustrative.jpeg" alt="GoldenRetriever logo" class="gr-hero-logo" />
  <p class="gr-eyebrow">Companion examples for Retriever</p>
  <h1>Move from runtime concepts to robot-facing examples.</h1>
  <p class="gr-lede">GoldenRetriever is the examples and integration surface for the core <code>retriever</code> runtime: perception, memory, language, composition, notebooks, robotics typing, and selected simulator/visualization demos.</p>
  <div class="gr-action-grid">
    <a class="gr-action-card" href="examples/">
      <span>01</span>
      <strong>Start with examples</strong>
      <small>Perception -> memory -> language -> composition.</small>
    </a>
    <a class="gr-action-card" href="examples/simulation_and_visualization_v1/">
      <span>02</span>
      <strong>See visual demos</strong>
      <small>Webcam + Rerun, MuJoCo/TWIST2, and browser control surfaces.</small>
    </a>
    <a class="gr-action-card" href="robotics_typing_standard/">
      <span>03</span>
      <strong>Use typed payloads</strong>
      <small>Robot payload contracts and data/event stream profiles.</small>
    </a>
  </div>
</div>

GoldenRetriever is intentionally separate from the core runtime docs. Use the core docs for `Flow`, clocks, sync policies, IR, and backend execution. Use this site when you want runnable robot-facing examples built on those primitives.

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
    <p>Rerun, web UI, and simulator lanes for richer demos.</p>
    <code>pixi run -e torch demo-webcam-rerun</code>
  </a>
</div>

## First Commands

=== "Concise ladder"

    ```bash
    pixi run -e golden-local demo-perception-detection-flow
    pixi run -e golden-local demo-memory-belief-flow
    pixi run -e golden-local demo-language-caption-plan
    pixi run -e golden-local demo-language-grounded-reference
    pixi run -e golden-local demo-composable-pipelines
    ```

=== "Visualization"

    ```bash
    pixi run -e torch demo-webcam-rerun
    pixi run -e torch demo-twist2-rerun
    ```

=== "Typing"

    ```bash
    pixi run demo-robotics-typing-catalog
    pixi run demo-robotics-typing-contract
    pixi run demo-robotics-typing-boundary
    ```

## Public Boundary

- Core runtime API details belong in `openretriever/retriever`.
- GoldenRetriever carries examples, tutorials, notebooks, robotics typing, and integration lanes.
- Heavy optional examples should stay clearly optional and mock-first where possible.
- Removed or stale experimental prototypes should not be presented as the main path.
