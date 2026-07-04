# OpenPI pi0.5 Policy Lane

> **Status: experimental / target integration.** Only `--mode mock` runs today
> (deterministic, no model, exercised by `pixi run demo-pi05-mock`). `--mode
> remote` needs a live openpi policy server, and `--mode hub` needs the
> `openretriever/pi05-policy` module published (see the packaging design below).
> Treat this lane as the worked design for the first Hub extension module, not a
> shipping integration yet.

Wire [Physical Intelligence's openpi](https://github.com/Physical-Intelligence/openpi)
pi0.5 vision-language-action model into a Retriever pipeline as an ordinary
`Flow[PolicyObservation, ActionChunk]` — and, as the target end state, load it
straight from the **Retriever Hub**.

## Run it

```bash
# Mock mode — deterministic, no model, what CI exercises:
pixi run demo-pi05-mock

# Against a live pi0.5 served by openpi (model runs on a GPU box):
pixi run python -m examples.advanced.openpi_policy.app --mode remote --host <gpu-box> --port 8000

# From the Retriever Hub (once the module below is published):
pixi run python -m examples.advanced.openpi_policy.app --mode hub
```

Expected result (mock): 8 lines like
`[mock] chunk horizon=10 dof=7 first_action=[ 0.084 ...]` — one action chunk
per observation frame.

## Why this shape

pi0.5 plans a *chunk* of future actions per inference call. That maps cleanly
onto Retriever's contract: the policy is a `Trigger("image")` flow that emits
an `ActionChunk` payload; a downstream controller can consume it with
`Latest()` (always act on the newest plan) or `Events()` (execute every
chunk). The heavy model never needs to live in this repo — openpi's
websocket server (`scripts/serve_policy.py` in the openpi repo) runs the
checkpoint (e.g. `pi05_droid`, `pi05_base` from `gs://openpi-assets`), and
the local flow only needs the lightweight `openpi-client` package.

## Hub packaging design (target integration)

Retriever Hub loads one module per git repository: the repo's
`pyproject.toml` declares `[tool.retriever.module]` exports, and an entry in
`openretriever/hub-index` makes it discoverable (see the core docs:
*Ecosystem → Publishing*).

**Recommended: a dedicated `openretriever/pi05-policy` repo.**

- Exports (all sharing this lane's typed contract):
  - `Pi05Policy` — remote/websocket flow (light deps: `openpi-client`, numpy)
  - `MockPi05Policy` — deterministic mock for CI and offline development
  - `Pi05LocalPolicy` — optional, imports full `openpi` for on-box inference
- Hub loads modules by repo tarball, so heavy extras stay optional and the
  import stays side-effect-free (no weights download at import).
- Index entry, `modules/openretriever/pi05-policy.toml` in
  `openretriever/hub-index`:

```toml
[module]
repo = "https://github.com/openretriever/pi05-policy"
description = "pi0.5 (openpi) policy flows: remote client, local, and mock"
author = "openretriever"
license = "Apache-2.0"
tags = ["policy", "vla", "manipulation", "openpi"]
```

- Usage anywhere, including this example's `--mode hub`:

```python
from retriever import hub
Pi05Policy = hub.use("openretriever/pi05-policy:Pi05Policy")
policy = Pi05Policy(host="gpu-box", port=8000) @ Trigger("image")
```

**Alternative considered: keep the module inside this (golden-retriever)
repo.** The Hub currently treats the repo root as the module root, so Golden
itself would have to become the hub module — dragging its full dependency
surface into `[tool.retriever.module]` dependency checks. Workable, but a
dedicated repo keeps the module small, versionable, and honest about its
dependencies. If in-repo modules become common, the cleaner fix is adding
subdirectory support to the Hub (`[module] repo = ... , path = "modules/pi05"`)
— a small core feature worth considering.

## References

- openpi repository: <https://github.com/Physical-Intelligence/openpi>
  (pi0, pi0-FAST, pi0.5 checkpoints; serving via `scripts/serve_policy.py`;
  `openpi-client` for remote inference)
- *π0.5: a Vision-Language-Action Model with Open-World Generalization*,
  Physical Intelligence, 2025. <https://www.physicalintelligence.company/blog/pi05>
- Retriever Hub publishing guide: core docs, *Ecosystem → Publishing*.
