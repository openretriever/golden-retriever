# AGENTS.md — GoldenRetriever, for coding agents

This file is the entry point for AI coding agents (and humans who read like
them). Exact commands over prose; complete snippets over fragments; stated
expected output over implication. If something here drifts from the repo,
fix the drift, not the reader.

## What this repo is

GoldenRetriever is the applied companion to the Retriever runtime
(`retriever-core`): runnable perception / memory / language / simulation /
visualization examples, plus `retriever_typing` — an applied
robotics/planning type pack. The pack declares a Retriever Hub module
manifest now; public `hub.use(...)` loading becomes usable after the
public `retriever-core` release and GitHub visibility cutover. Until that
cutover, Pixi owns the runtime dependency details. The local build metadata
remains for wheel smoke tests, but a public GoldenRetriever PyPI wheel is not on the
first launch path.

- Python: **3.11+**. License: Apache-2.0.
- Docs site: <https://golden.retriever.build/>
- Core runtime docs: <https://retriever.build/> (agents: also
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
pixi run test                                      # full GoldenRetriever suite
pixi run -e golden-retriever demo-perception-detection-flow
pixi run demo-robosuite-mock                       # simulation lane, mock mode
pixi run demo-pipeline-html-viz                     # self-contained IR/HTML visualization
pixi run public-surface-check                     # current-tree public surface guardrail
pixi run demo-pi05-mock                            # pi0.5 policy lane, mock mode
pixi run -e docs docs-build                        # Starlight build
pixi run build                                     # local wheel artifact; not first-launch PyPI path
```

Environments: `default` and `golden-retriever` currently resolve the runtime
through the Pixi dependency table until `retriever-core` is published;
`golden-retriever` adds GoldenRetriever extras. For prerelease core validation, set
`RETRIEVER_CORE_SRC=<core-repo>/src` before running
`pixi run test`. Heavier lanes (`torch`, `vlm`, `twist2`, ...) exist for
specific examples — check `[environments]` in `pixi.toml` before assuming
one.

## Layout

| Area | Path |
| --- | --- |
| Applied payloads (Hub type pack / local wheel) | `src/retriever_typing/` |
| Arrow serialization (round-trip tested) | `src/retriever_typing/conversions.py` |
| Spatial standard (re-exports `retriever.types.spatial` — same classes) | `src/retriever_typing/v1.py` |
| Example lanes (learning ladder) | `examples/advanced/` |
| Heavier prototypes | `examples/experimental/` |
| Tests | `tests/typing/`, `tests/examples/` |
| Docs (Starlight -> golden.retriever.build) | `docs-site/` |

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
- Treat projects, repositories, datasets, features, workspaces, hosts, and
  research directions learned only from conversations or the local environment
  as confidential unless they are already public here or the user explicitly
  approves disclosure.
- Never describe confidential work merely to say that it is excluded, private,
  downstream, or owned elsewhere. Omit the reference entirely and describe only
  the public interface that exists in this repository.
- Before committing or pushing, scan both the proposed tree and the branch
  history for confidential names and descriptions; checking only the final file
  contents is insufficient.
- Do not commit conversation transcripts, prompt or tool logs, agent names or
  identities, model or provider metadata, task or worktree IDs, or agent
  scratch files. Public code, docs, commit messages, and artifacts should
  describe project behavior and source provenance, not which agent produced
  them.
- Do not commit machine-specific absolute paths, local account names,
  temporary-file locations, credentials, access codes, or private dataset
  locations. Use documented environment variables and repository-relative
  paths instead.
- Keep these rules canonical in `AGENTS.md`. Files for other coding tools may
  point here, but must not maintain a separate copy that can drift.

## Forbidden actions

- **DO NOT** execute `git checkout .` or any `git checkout` command that
  discards changes without explicit user permission.
- **DO NOT** execute `git clean` or `git reset --hard` without explicit
  user permission.
- **DO NOT** execute `rm -rf` on non-trivial directories without explicit
  plan approval.

## Commits and pull requests

**Never put any sign of agent authorship into git history or anything outward-facing.**

Do not add, and strip if present:

- `Co-Authored-By:` trailers naming Claude, Codex, Anthropic, OpenAI or any model
- `Claude-Session:`, `Generated with`, `🤖` or similar markers
- Any mention of an assistant in commit messages, PR titles and bodies, code
  comments, docs, or issue comments

Commits are authored by the human running the session, full stop. Before every
commit, push or PR, check the message for these markers and remove them. When
amending existing work, check the whole branch, not just `HEAD`.

The one exception: commits already published by someone else. Do not rewrite
another person's pushed history to strip their trailers — that breaks merges
against their branch and does not remove the trailers from the remote anyway.
Tell them instead.
