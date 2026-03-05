# Branch Status and PR Readiness (2026-03-05)

## Scope
This note tracks which branches are intended for PRs into `origin/main` and which are explicitly excluded from this merge wave.

## Merge Target
- Target branch: `origin/main`

## Active Origin Branches (Major Unmerged Work)

### `origin/feat/robotics-typing-v1`
- Purpose: robotics typing public surface, registry, migration updates, docs/examples/tests.
- Current state (before cleanup push): ahead `33`, behind `15` vs `origin/main`.
- Include policy:
  - `src/golden_retriever/robotics_typing/*`
  - `src/golden_retriever/types/*` public-surface updates
  - `docs/robotics_typing_standard/*`
  - `examples/advanced/robotics_typing_standard/*`
  - robotics typing tests
- Exclude policy:
  - `experiments/determinism*`
  - `examples/tutorial_mirror/*`
- Action: force-update this remote branch from clean rewrite line.

### `origin/dev/add-examples-202602`
- Purpose: notebook + examples workflow additions (including jupytext pairing workflow).
- State: ahead `5`, behind `0` vs `origin/main`.
- Policy: keep independent PR line; do not mix with typing or experiments branches.

### `origin/experiments/determinism`
- Purpose: determinism/backprop experiments and result artifacts.
- State: ahead `26`, behind `0` vs `origin/main`.
- Policy: keep as experiments-only PR line; do not mix with typing or examples PRs.

### `origin/feature/chaotic_control`
- Purpose: separate chaotic control idea branch.
- State: ahead `1`, behind `31` vs `origin/main`.
- Policy: excluded from this merge wave.

## Mirror Remote Classification
`mirror/*` branches are treated as sync/archive sources, not direct PR merge candidates into `origin/main` in this wave.

## Backup / Audit Safety
- Preserve backup branch history for typing cleanup traceability:
  - `backup/feat-robotics-typing-v1-before-clean-20260305`
