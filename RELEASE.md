# GoldenRetriever Release Checklist

GoldenRetriever is the public maintained reference examples repository for Retriever examples, robot type packs, simulator/visualization lanes, and future Retriever Hub packs.

## Required Validation

Run these before a public launch or docs promotion:

```bash
pixi run test
pixi run -e docs docs-build
pixi run -e golden-local demo-perception-detection-flow
pixi run public-surface-check
```

Build the optional local/future wheel artifact separately when checking package
contents:

```bash
pixi run build
```

The same checks are wired in `.github/workflows/ci.yml`. `public-surface-check` is the guardrail that keeps removed legacy experimental lanes out of the release tree while retaining the robosuite mock and HTML visualization examples.

Before the public Hub index and `retriever-core==0.0.1` wheel are live, run the Hub-pack smoke against a local core checkout when validating the final cutover:

```bash
RETRIEVER_CORE_SRC=<core-repo>/src PYTHONPATH=<core-repo>/src:. pixi run -e golden-local demo-golden-hub-pack
```

Do not add this Hub-pack smoke as a required remote CI gate until the public runtime wheel and Hub index are available to CI.



## Built-Artifact Local Preview

Use this when reviewing exactly what the Golden Pages project will receive:

```bash
pixi run -e docs docs-build
python3 -m http.server 8782 --bind 127.0.0.1 --directory site
```

Open `http://127.0.0.1:8782/`, `http://127.0.0.1:8782/hub/`, `http://127.0.0.1:8782/llms.txt`, and `http://127.0.0.1:8782/robots.txt`. The home page should show `Golden Examples for Retriever`, `Recommended Route`, `Command Matrix`, `Example Result Shapes`, and the Retriever ecosystem map.

## Post-Deploy Content Check

After deploying the Golden docs site, verify the live page reflects the reference-catalog framing rather than stale Hub-module wording:

```bash
html=$(curl -fsSL https://retriever-space.pages.dev/)
printf '%s' "$html" | grep -q 'Golden Examples for Retriever'
printf '%s' "$html" | grep -q 'Recommended Route'
printf '%s' "$html" | grep -q 'Example Result Shapes'
printf '%s' "$html" | grep -q 'What Belongs Where'
legacy_title='GoldenRetriever Hub'' Module'
legacy_subtitle='first app''lied robotics Hub'' module'
! printf '%s' "$html" | grep -q "$legacy_title"
! printf '%s' "$html" | grep -q "$legacy_subtitle"
curl -fsSL https://retriever-space.pages.dev/robots.txt | grep -q 'Sitemap: https://retriever-space.pages.dev/sitemap.xml'
curl -fsSL https://retriever-space.pages.dev/robots.txt | grep -q 'Agent map: https://retriever-space.pages.dev/llms.txt'
curl -fsSL https://retriever-space.pages.dev/llms.txt | grep -q 'Golden Examples for Retriever'
```

If a custom Golden domain is bound later, run the same checks against that hostname before advertising it.

## GitHub Settings

Before making the repository public:

- Confirm the default branch is `main`.
- Deploy the MkDocs build through the configured static hosting target.
- Confirm the repository URL is `https://github.com/openretriever/golden-retriever`.
- Confirm the hosted examples/docs URL is `https://retriever-space.pages.dev/` until custom domains are active.

## Package Boundary

First public launch does not require publishing a separate Golden PyPI package. Golden's robot-facing planning payloads are exposed through the Retriever Hub pack manifest in `pyproject.toml`, and any wheel remains an optional local/future artifact. If a wheel is published later, it should ship only the lightweight compatibility/type-pack surface; heavy examples, notebooks, benchmarks, generated outputs, and optional robot/model stacks remain source-checkout material or local artifacts.

## Runtime Dependency

When the public `retriever-core` distribution is published, update:

- `pixi.toml` bundled/runtime core dependencies,
- README setup text,
- docs setup text,
- clean-clone validation notes,
- `AGENTS.md` (describes the interim runtime dependency boundary),
- `examples/advanced/core_composition/README.md` (states the retriever-core
  end-state as current),
- `docs/examples/simulation_and_visualization_v1.md` (drop the `--no-deps`
  hedge on the robosuite install).
