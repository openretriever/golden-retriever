# GoldenRetriever Release Checklist

GoldenRetriever is the public companion repository for advanced examples, system integrations, and research prototypes built on the core Retriever runtime.

## Required Validation

Run these before a public launch, tag, or package publish:

```bash
pixi run test
pixi run -e docs docs-build
pixi run -e golden-local demo-perception-detection-flow
pixi run build
```

The same checks are wired in `.github/workflows/ci.yml`.

## GitHub Settings

Before making the repository public:

- Confirm the default branch is `main`.
- Enable GitHub Pages with source `GitHub Actions` so `.github/workflows/docs.yml` can deploy `site/`.
- Confirm the repository URL is `https://github.com/openretriever/golden-retriever`.

## Package Boundary

The wheel intentionally ships the lightweight `retriever_typing` package. Heavy examples, notebooks, benchmarks, generated outputs, and optional robot/model stacks remain source-checkout material or local artifacts.

## Runtime Dependency

Golden currently uses the temporary `debug-retriever` package for portable demos. After the real public `retriever` package is published, update:

- `pixi.toml` bundled/runtime core dependencies,
- README setup text,
- docs setup text,
- clean-clone validation notes.
