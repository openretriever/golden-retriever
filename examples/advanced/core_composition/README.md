
# Core Composition Examples

These examples exercise registry-backed pipeline composition surfaces.

## Runtime Requirement

Use the Golden demo environment so the example feature set and optional dependencies are available. During release preparation this environment uses the documented source-checkout runtime path; set `RETRIEVER_CORE_SRC` only when you intentionally want to validate against a different local core checkout. After `retriever-core` is published, the same import remains `retriever`.

```bash
pixi install -e golden-local
pixi run -e golden-local demo-composable-pipelines
```

You can inspect the runtime package being used with:

```bash
pixi run -e golden-local python -c "import retriever; print(retriever.__file__)"
```

## Examples

- `composable_pipelines.py`: register a small pipeline, override one internal stage, inject surfaced inputs, and wrap the pipeline back into a larger graph with `build_pipeline_flow(...)`. The payloads stay on one small shared type vocabulary; the example is about changing structure, not inventing pipeline-specific schema classes.
