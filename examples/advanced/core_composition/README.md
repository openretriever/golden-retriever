
# Core Composition Examples

These examples exercise registry-backed pipeline composition surfaces.

## Runtime Requirement

Use the Golden demo environment so the example feature set is available. During prerelease this environment may still consume the temporary `debug-retriever` runtime; after `retriever-core==0.0.1` is public it should validate against `retriever-core`.

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
