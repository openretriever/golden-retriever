# GoldenRetriever Release Checklist

GoldenRetriever is the public companion repository for advanced examples, system integrations, and research prototypes built on the core Retriever runtime.

## Required Validation

Run these before a public launch or docs promotion:

```bash
pixi run test
pixi run -e docs docs-build
pixi run -e golden-local demo-perception-detection-flow
```

Build the optional local/future wheel artifact separately when checking package
contents:

```bash
pixi run build
```

The same checks are wired in `.github/workflows/ci.yml` where configured.

## GitHub Settings

Before making the repository public:

- Confirm the default branch is `main`.
- Deploy the MkDocs build through the configured Cloudflare Pages project `retriever-space`.
- Confirm the repository URL is `https://github.com/openretriever/golden-retriever`.
- Confirm the hosted examples/docs URL is `https://retriever-space.pages.dev/` until custom domains are active.

## Package Boundary

First public launch does not require publishing a `retriever-golden` PyPI package. Golden's applied robotics/planning payloads are exposed through the Retriever Hub module manifest in `pyproject.toml`, and the wheel remains an optional local/future artifact. If a wheel is published later, it should ship only the lightweight `retriever_typing` package; heavy examples, notebooks, benchmarks, generated outputs, and optional robot/model stacks remain source-checkout material or local artifacts.

## Runtime Dependency

When the public `retriever-core` distribution is published, update:

- `pixi.toml` bundled/runtime core dependencies,
- README setup text,
- docs setup text,
- clean-clone validation notes.
