# GoldenRetriever Docs

GoldenRetriever keeps the public docs small and topic-based.

Core runtime source for now: Golden demos use the temporary `debug-retriever` PyPI package until the public `openretriever` distribution is published.

## Start Here

- `docs/examples/README.md`: practical example guides built around runnable launch points.
- `docs/robotics_typing_standard/README.md`: typed payload, schema, and data-profile guides.

## Recommended Reading Order

1. `docs/examples/README.md`
2. `docs/examples/perception_and_memory_v1.md`
3. `docs/examples/pipeline_composition_v1.md`
4. `docs/robotics_typing_standard/README.md`


## Hostable Site

`mkdocs.yml` defines a small public site map for these docs. The configured public target is `openretriever/golden-retriever`, with the core runtime docs hosted separately by the main `retriever` repo.

```bash
pixi run -e docs docs-build
pixi run -e docs docs-serve
```

For a quick release check from a clean clone:

```bash
pixi run -e docs docs-build
pixi run -e golden-local test
pixi run -e golden-local demo-perception-detection-flow
pixi run build
```
