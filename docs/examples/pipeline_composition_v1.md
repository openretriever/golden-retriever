# Core Composition Surfaces v1

This guide covers the newer registry-backed pipeline composition surfaces used by the Golden examples. Keep the examples on the packaged runtime path; until `retriever-core==0.0.1` is published, use the configured Golden environment or a local `RETRIEVER_CORE_SRC` checkout for unreleased runtime changes.

## Runtime check

```bash
pixi install -e golden-local
pixi run -e golden-local python -c "import retriever; print(retriever.__file__)"
```

The printed path should resolve to the installed runtime package for this environment. In prerelease environments it may show the temporary debug package; after public launch it should resolve to `retriever-core`.

## First runnable example

```bash
pixi run -e golden-local demo-composable-pipelines
```

That example, `examples/advanced/core_composition/composable_pipelines.py`, demonstrates three capabilities that are not shown in Golden's older `functional_wiring/` folder:

1. register a pipeline by name with explicit surfaced input/output selectors
2. replace one named internal stage after the pipeline is built
3. wrap the registered pipeline back into a larger graph via `build_pipeline_flow(...)`

## Why this is separate from `functional_wiring/`

Golden's existing `functional_wiring/` examples are still the right starting point for explicit graph construction. The new composition surface sits one level higher:

- `functional_wiring/` keeps subgraph construction explicit and local
- `core_composition/` exercises registry-backed reuse and pipeline-as-flow composition

Use the older folder first if you are learning the runtime. Use the newer folder when you want reusable named pipeline building blocks.

Do not treat this layer as a license to invent pipeline-specific envelope types. The preferred pattern is still: shared primitive payloads first, composite `Flow[...]` typing for local structure, named envelopes only when the boundary is reused and semantically stable.

## Related examples

- `examples/advanced/perception_debug/detection_window_stats.py`: add a temporal aggregation stage to a deterministic perception pipeline
- `examples/advanced/state_management/stateful_replanning.py`: add internal planner memory and change-only event emission
- `examples/advanced/functional_wiring/perception_belief_control_pipeline.py`: compose a belief stage into downstream control without relying on the registry-backed layer

## Notebook version

If you want the same composition path in notebook form, build and run the Hub-first notebook:

```bash
pixi run notebook-to-ipynb-hub
pixi install -e golden-local
pixi run -e golden-local demo-hub-notebook-source
```

The notebook lives at `notebooks/src/hub_demo.py` and stays parameterized by environment variables so the repo does not hardcode any private or organization-specific published module refs.
