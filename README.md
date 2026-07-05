<div align="center">
  <a href="https://retriever-space.pages.dev/"><img width="180px" height="auto" src="assets/retriever-illustrative.jpeg" alt="GoldenRetriever logo"></a>
</div>

# 🐕 <span style="background: linear-gradient(45deg, #e96443 0%, #904e95 25%, #e65c00 50%, #f9d423 75%, #fc00ff 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; font-weight: bold; font-size: 1.1em;">**GoldenRetriever**</span>

## **Golden Reference Examples and Type Packs for Retriever**

<div align="center">

<p>GoldenRetriever is the maintained reference examples layer for the core <code>retriever</code> runtime: maintained examples, reusable robot-facing type packs, simulator/visualization lanes, and candidates for Retriever Hub packs.</p>

<p>
  <a href="https://github.com/openretriever/retriever"><img alt="Core runtime" src="https://img.shields.io/badge/Core-runtime-111827?style=for-the-badge"></a>
  <a href="https://openretriever-docs.pages.dev/"><img alt="Core docs" src="https://img.shields.io/badge/Core-docs-0f766e?style=for-the-badge"></a>
  <a href="https://retriever-space.pages.dev/"><img alt="Golden docs" src="https://img.shields.io/badge/Golden-docs-92400e?style=for-the-badge"></a>
  <a href="https://openretriever.org/"><img alt="Website" src="https://img.shields.io/badge/Website-openretriever.org-111827?style=for-the-badge"></a>
  <a href="https://retriever-space.pages.dev/llms.txt"><img alt="Golden agent map" src="https://img.shields.io/badge/Agent-map-f97316?style=for-the-badge"></a>
</p>

</div>

---

GoldenRetriever is the maintained reference examples layer for Retriever. Keep runtime mechanics in the [Retriever core docs](https://openretriever-docs.pages.dev/); use this repo for maintained robot-facing examples, reusable type packs loaded through Retriever Hub, notebooks, simulation, visualization, and promotion candidates for future Hub-distributed packs.

## Project Boundary

- The core runtime distribution is `retriever-core`; Python imports remain `retriever`.
- GoldenRetriever is not a second runtime package. It is the examples-and-packs layer on top of Retriever: examples, robot type packs, integration patterns, notebooks, simulator wrappers, and visualization lanes.
- The current manifest-declared surface is the robot-facing type pack plus conversion helpers declared in `pyproject.toml` and loaded through Retriever Hub. Source examples become Hub packs only after they are import-safe, versioned, smoke-tested, and documented.

## Setup

```bash
pixi install
```

For the main Golden example environment:

```bash
pixi install -e golden-local
pixi run -e golden-local python -c "import retriever; print('retriever import OK')"
```

`golden-local` and `golden-perception` are retained as stable launch environments for existing demo commands.

## Recommended Launch Points

Use Golden after the core runtime quickstart. The path is: learn `Flow` and visual debugging in core, prove the Retriever Hub extension boundary, then walk the robot-facing example ladder.

If you are new to Retriever, start with the core visual quickstart in the core repository first:

- Core docs: https://openretriever-docs.pages.dev/getting-started/visual-quickstart/
- Core source: https://github.com/openretriever/retriever

That core quickstart starts with `pixi run demo-webcam-detection-mock` from `openretriever/retriever`, then switches to `pixi run demo-webcam-detection` for live webcam/Rerun; those are not GoldenRetriever tasks. Then use Golden for robot-facing Hub-pack proof and example families. First prove the Retriever Hub pack boundary, then follow the concise perception -> memory -> language -> composition ladder:

```bash
pixi run demo-golden-hub-pack
pixi run -e golden-local demo-perception-detection-flow
pixi run -e golden-local demo-memory-belief-flow
pixi run -e golden-local demo-language-caption-plan
pixi run -e golden-local demo-language-grounded-reference
pixi run -e golden-local demo-composable-pipelines
```

Then use mock-safe topic commands as needed:

```bash
pixi run demo-robotics-typing-catalog
pixi run -e golden-local demo-perception-segmentation-flow
pixi run -e golden-local demo-perception-pointing-flow
pixi run -e golden-local demo-memory-dropout-flow
pixi run -e golden-local demo-memory-pointing-flow
pixi run demo-perception-record
pixi run demo-perception-replay
pixi run -e golden-local demo-detection-window-stats
pixi run demo-perception-replay-to-belief
pixi run demo-perception-belief-control
pixi run demo-multi-agent-communication
```

Model-backed perception lanes are optional and should be run only after the mock-safe ladder is green and the required credentials/dependencies are configured:

```bash
pixi run -e golden-perception demo-gemini-detection-flow
pixi run -e golden-perception demo-belief-from-real-detections
```

Some heavier optional Pixi environments still resolve demo-only dependencies from Git repositories. Those dependencies are intentionally kept in `pixi.toml`, not the Python package metadata, so default docs and concise examples remain installable without cloning extra research stacks.


## How Golden Fits The Retriever Ecosystem

| Layer | What lives there | Public route |
| --- | --- | --- |
| Retriever home | Product-level explanation and routing. | [openretriever.org](https://openretriever.org/) |
| Core runtime docs | `Flow`, `Pipeline`, clocks, sync policies, runtime execution, IR, Hub loader, install, and visual quickstart. | [Core docs](https://openretriever-docs.pages.dev/) |
| GoldenRetriever | Maintained robot-facing examples, type packs, simulator wrappers, visualization lanes, notebooks, and Hub pack candidates. | [Golden docs](https://retriever-space.pages.dev/) |
| Core source | Core Retriever implementation. | [openretriever/retriever](https://github.com/openretriever/retriever) |
| Golden source | Robot-facing examples, type packs, and simulator/visualization lanes. | [openretriever/golden-retriever](https://github.com/openretriever/golden-retriever) |

Golden pages should answer one question: "What reusable Golden pack, payload, or example should I run next after I understand the core Retriever runtime?"

## Repository Layout

- `examples/advanced`: runnable advanced demos with concrete launch points. Start with `examples/advanced/README.md`.
- `docs/examples`: public example-guide articles. Start with `docs/examples/README.md`.
- `src/retriever_typing`: typed robotics and event/data helpers used by advanced demos and exposed through the Retriever Hub pack manifest.
- `docs/robotics_typing_standard`: typed payload and data-profile notes for this repo.
- `notebooks`: git-friendly notebook sources and generated notebook artifacts. Start with `notebooks/README.md`.
- `docs`: public topic-based docs; `mkdocs.yml` provides a hostable site map.

## Example Families

- `examples/advanced/perception_examples`: concise detection, segmentation, and pointing flows over one shared synthetic scene.
- `examples/advanced/memory_examples`: concise belief and remembered-pointing flows over the same perception payloads.
- `examples/advanced/language_examples`: caption, grounding, and primitive plan-text examples.
- `examples/advanced/perception_debug`: synthetic perception, windowed stats, MCAP recording, and replay.
- `examples/advanced/real_perception`: optional model-backed perception lanes with mock-first defaults.
- `examples/advanced/real_memory`: optional real/mock memory flows built on detection and belief payloads.
- `examples/advanced/state_management`: state, reset behavior, and memory-oriented flows.
- `examples/advanced/functional_wiring`: flow composition, fan-in/fan-out, staged builders, and sync policies.
- `examples/advanced/core_composition`: registry-backed pipeline composition surfaces.
- `examples/advanced/multi_agent_communication`: a compact coordination/composition example.
- `examples/advanced/webcam_rerun`: webcam/model/Rerun visualization.
- `examples/advanced/twist2_simulation`: MuJoCo/TWIST2 simulator integration.
- `examples/advanced/mujoco_manipulation`: MuJoCo manipulation with Rerun logging.
- `examples/advanced/robosuite_lift`: mock-safe robosuite Lift smoke demo.
- `examples/advanced/hierarchical_physics_demo`: physics demos with HTML pipeline visualization.

For the end-to-end perception -> memory -> composition walkthrough, see `docs/examples/perception_and_memory_v1.md`. For registry-backed composition surfaces, continue with `docs/examples/pipeline_composition_v1.md`.

Design notes that are not runnable example families stay separate from the main path. `examples/advanced/closed_loop_planning` is an extracted pattern note, not a first-run demo.

## Simulation And Visualization

Use these optional lanes after the concise examples are clear:

```bash
pixi run -e torch demo-webcam-rerun
pixi run -e twist2 demo-twist2-rerun
pixi run demo-robosuite-mock
pixi run demo-pipeline-html-viz
pixi run public-surface-check
```

The current visual lanes cover webcam/Rerun perception, MuJoCo/TWIST2 simulation, MuJoCo manipulation, a mock-safe robosuite Lift smoke demo, hierarchical physics with HTML pipeline visualization, and the promoted pipeline HTML utility. `public-surface-check` also runs short Hub, robosuite mock, and HTML visualization smokes. See `docs/examples/simulation_and_visualization_v1.md` for the public guide.

## Typed Payload Demos

```bash
pixi run demo-robotics-typing-catalog
pixi run demo-robotics-typing-contract
pixi run demo-robotics-typing-boundary
```

For runnable type/data examples, start with `examples/advanced/robotics_typing_standard/README.md`.

## Notebook Workflow

```bash
pixi run notebook-to-ipynb-demo
pixi run notebook-to-ipynb-hub
```

These tasks regenerate the git-friendly Jupytext notebooks. `retriever_demo` is a mechanics notebook for the packaged Golden environment. `hub_demo` is the Hub-first notebook and is meant to be run from the Golden demo environment:

```bash
pixi install -e golden-local
pixi run -e golden-local demo-hub-notebook-source
```

The Hub notebook reads Hub refs from environment variables instead of hardcoding private or organization-specific pack names.

## Documentation Site

The docs are structured as the Golden examples layer linked from the Retriever landing page and core runtime docs.

```bash
pixi run -e docs docs-build
pixi run -e docs docs-serve
```

Keep Golden docs example-first: concise perception, memory, language, composition, and robotics typing guides belong here; core runtime API details belong in the main `retriever` repo.

## Validation

Use these checks before opening a release branch or publishing docs:

```bash
pixi run -e docs docs-build
pixi run -e golden-local test
pixi run -e golden-local demo-perception-detection-flow
pixi run demo-golden-hub-pack
pixi run demo-robosuite-mock
pixi run demo-pipeline-html-viz
pixi run public-surface-check
pixi run build
```

See `RELEASE.md` for launch/docs validation and the optional package boundary checklist.

## Relationship To Core Retriever

- PyPI distribution: `retriever-core`
- Python import: `retriever`
- Retriever landing: `https://openretriever.org/`
- Runtime source: `https://github.com/openretriever/retriever`
- Runtime docs: `https://openretriever-docs.pages.dev/`
- Golden source: `https://github.com/openretriever/golden-retriever`
- Golden docs: `https://retriever-space.pages.dev/`

## Contributing And License

See `CONTRIBUTING.md` for the contribution workflow and `SECURITY.md` for private vulnerability reporting. GoldenRetriever is licensed under the Apache License 2.0; see `LICENSE`.
