# GoldenRetriever Docs

GoldenRetriever keeps the public docs small and topic-based.

Core runtime source for now: default demos use the temporary `debug-retriever` PyPI package; local editable demos use a sibling `../retriever` checkout.

## Start Here

- `docs/examples/README.md`: practical example guides built around runnable launch points.
- `docs/robotics_typing_standard/README.md`: typed payload, schema, and data-profile guides.

## Recommended Reading Order

1. `docs/examples/README.md`
2. `docs/examples/perception_and_memory_v1.md`
3. `docs/examples/pipeline_composition_v1.md`
4. `docs/robotics_typing_standard/README.md`


## Hostable Site

`mkdocs.yml` defines a small public site map for these docs. Use it for local previews now and replace `site_url`/repo links when the public hosting location is final.

```bash
pixi run -e docs docs-build
pixi run -e docs docs-serve
```
