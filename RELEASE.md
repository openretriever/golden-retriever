# GoldenRetriever Release Checklist

GoldenRetriever is the public maintained reference examples repository for Retriever examples, robot payload references, simulator/visualization lanes, and future Retriever Hub packs.

## Required Validation

Run these before a public launch or docs promotion:

```bash
pixi run test
pixi run -e docs docs-build
pixi run -e golden-retriever demo-perception-detection-flow
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
RETRIEVER_CORE_SRC=<core-repo>/src PYTHONPATH=<core-repo>/src:. pixi run -e golden-retriever demo-golden-hub-pack
```

Do not add this Hub-pack smoke as a required remote CI gate until the public runtime wheel and Hub index are available to CI.



## Built-Artifact Local Preview

Use this when reviewing exactly what the Golden Pages project will receive:

```bash
pixi run -e docs docs-build
python3 -m http.server 8782 --bind 127.0.0.1 --directory docs-site/dist
```

Open `http://127.0.0.1:8782/`, `http://127.0.0.1:8782/examples/`, `http://127.0.0.1:8782/robot-payloads/`, `http://127.0.0.1:8782/llms.txt`, and `http://127.0.0.1:8782/robots.txt`. The home page should show `Golden examples start where the core quickstart ends`, `First Results To Recognize`, `Recommended Route`, and the Retriever surface map.

## Post-Deploy Content Check

After deploying the Golden docs site, verify the live page reflects the applied-reference framing rather than stale Hub-module wording:

```bash
html=$(curl -fsSL https://retriever-space.pages.dev/)
printf '%s' "$html" | grep -q 'Golden examples start where the core quickstart ends'
printf '%s' "$html" | grep -q 'First Results To Recognize'
printf '%s' "$html" | grep -q 'How The Retriever Surfaces Fit'
curl -fsSL https://retriever-space.pages.dev/robot-payloads/ | grep -q 'Robot Payload Reference'
curl -fsSL https://retriever-space.pages.dev/robot-payloads/type-catalog/ | grep -q 'Robot Payload Selection'
curl -fsSL https://retriever-space.pages.dev/robots.txt | grep -q 'Sitemap: https://retriever-space.pages.dev/sitemap-index.xml'
curl -fsSL https://retriever-space.pages.dev/robots.txt | grep -q 'LLM map: https://retriever-space.pages.dev/llms.txt'
curl -fsSL https://retriever-space.pages.dev/llms.txt | grep -q 'Golden Retriever'
```

If a custom Golden domain is bound later, run the same checks against that hostname before advertising it.

## GitHub Settings

Before making the repository public:

- Confirm the default branch is `main`.
- Deploy the Starlight docs build through the configured static hosting target.
- Confirm the repository URL is `https://github.com/openretriever/golden-retriever`.
- Confirm the hosted examples/docs URL is `https://retriever-space.pages.dev/` until custom domains are active.

## Package Boundary

First public launch does not require publishing a separate Golden PyPI package. Golden's robot-facing planning payloads are exposed through the Retriever Hub module manifest, and any wheel remains an optional local/future artifact. If a wheel is published later, it should ship only the lightweight compatibility/payload-pack surface; heavy examples, notebooks, benchmarks, generated outputs, and optional robot/model stacks remain source-checkout material or local artifacts.

## Runtime Dependency

When the public `retriever-core` distribution is published, update:

- `pixi.toml` bundled/runtime core dependencies,
- README setup text,
- docs setup text,
- clean-clone validation notes,
- repository guidance for coding agents and runtime dependency boundaries,
- `examples/advanced/core_composition/README.md` (states the retriever-core
  end-state as current),
- `docs/examples/simulation_and_visualization_v1.md` (drop the `--no-deps`
  hedge on the robosuite install).
