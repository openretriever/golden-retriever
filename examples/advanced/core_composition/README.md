# Core Composition Examples

These examples exercise the newer registry-backed pipeline composition surfaces from the current `retriever-mirror` core.

## Local core requirement

Use the local editable Golden environment so `retriever` resolves to the sibling `../retriever-mirror` checkout:

```bash
pixi install -e golden-local
pixi run -e golden-local demo-composable-pipelines
```

## Examples

- `composable_pipelines.py`: register a small pipeline, override one internal stage, inject surfaced inputs, and then wrap the pipeline back into a larger graph with `build_pipeline_flow(...)`.
