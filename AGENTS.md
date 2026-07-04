# AGENTS.md — GoldenRetriever, for coding agents

This file is the entry point for AI coding agents (and humans who read like
them). Exact commands over prose; complete snippets over fragments; stated
expected output over implication. If something here drifts from the repo,
fix the drift, not the reader.

## What this repo is

GoldenRetriever is the applied companion to the Retriever runtime
(`retriever-core`): runnable perception / memory / language / simulation /
visualization examples, plus `retriever_typing` — a Hub-loadable applied
robotics/planning type pack. The local build metadata still uses the
`retriever-golden` package name, but a public PyPI wheel is not on the first
launch path.

- Python: **3.11+**. License: Apache-2.0.
- Docs site: <https://retriever-space.pages.dev/>
- Core runtime docs: <https://openretriever-docs.pages.dev/> (agents: also
  read `AGENTS.md` in the core repo and `/llms.txt` on the docs site).

## Documentation goals (keep these when editing anything)

1. Every example lane is mock-first: it must run without hardware, models,
   or GPUs, and say what its expected output is.
2. Key concepts stay explainable in under a minute; deeper theory links to
   the core docs (`concepts/lineage`).
3. **AI agents are first-class readers** — of docs and of example code.
   Keep examples import-safe, argument-documented, and deterministic in
   mock mode.

## Commands (verified)

```bash
pixi run test                                      # 52+ tests, <5s
pixi run -e golden-local demo-perception-detection-flow
pixi run demo-robosuite-mock                       # simulation lane, mock mode
pixi run demo-pi05-mock                            # pi0.5 policy lane, mock mode
pixi run -e docs docs-build                        # mkdocs --strict
pixi run build                                     # local retriever-golden wheel artifact
```

Environments: `default` bundles the core runtime; `golden-local` = local
runtime + golden extras; heavier lanes (`torch`, `vlm`, `twist2`, ...) exist
for specific examples — check `[environments]` in `pixi.toml` before
assuming one.

## Layout

| Area | Path |
| --- | --- |
| Applied payloads (Hub type pack / local wheel) | `src/retriever_typing/` |
| Arrow serialization (round-trip tested) | `src/retriever_typing/conversions.py` |
| Spatial standard (re-exports `retriever.types.spatial` — same classes) | `src/retriever_typing/v1.py` |
| Example lanes (learning ladder) | `examples/advanced/` |
| Heavier prototypes | `examples/experimental/` |
| Tests | `tests/typing/`, `tests/examples/` |
| Docs (mkdocs → retriever-space.pages.dev) | `docs/` |

## Pitfalls that actually bite

- Examples use the **current** runtime API: `@io` payloads, `Flow.step()`
  (never `run()`), mandatory `sync=` on `connect()`.
- Standard payload types live in `retriever.types.*` (the runtime);
  `retriever_typing.v1` re-exports them as the same class objects. Never
  redefine a standard type locally — type identity is the contract.
- `retriever_typing` uses lazy exports (`__getattr__` in `__init__.py`);
  add new public types to both `_TYPE_MODULES` and `__all__`.
- Arrow conversion lives in `conversions.py` only; round-trip every new
  type in `tests/typing/test_conversions_roundtrip.py`.
- Raw data under `logs/` is generated, never edited, never committed.

## Content rules

- Do not introduce references to private infrastructure, internal
  hostnames, or unpublished project names into this repository.

## Forbidden actions

- **DO NOT** execute `git checkout .` or any `git checkout` command that
  discards changes without explicit user permission.
- **DO NOT** execute `git clean` or `git reset --hard` without explicit
  user permission.
- **DO NOT** execute `rm -rf` on non-trivial directories without explicit
  plan approval.
