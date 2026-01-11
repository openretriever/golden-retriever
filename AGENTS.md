# Agent Rules

## Critical Workflow Rules
- **Fully iterate and debug locally**: Make sure to validat code changes by running them yourself before notifying the user. Do not rely on the user to debug your code.
- **Privacy**: Do not put private info in commit messages.

## Repo Context
- This repo is the **Retriever runtime/core**. System-level demos/integrations belong in a separate golden-retriever repo.
- Prefer **Pipeline + FlowHandle** APIs for new examples and docs.

## Examples + Docs Guidelines
- Avoid ENV-driven configs in examples; prefer `argparse` (or `typer`/`click` if needed).
- Keep examples deterministic by default; hardware should be opt-in via flags.
- Include a short **Run:** snippet at the top of example files.
- One concept per example; keep dependencies light.

## Runtime Constraints / Lessons
- Backend execution reconstructs flows from IR:
  - If a Flow needs constructor args, implement `Flow.init_config()` (and optionally `Flow.from_init_config()`).
  - Otherwise, keep flows default-constructible.
- Triggered flows may run before all inputs arrive; guard against `None` in `Flow.run()`.
- For debugging, use `Pipeline.step()` and call `Pipeline.close_stepper()` to release resources.

## Agent Notes
- If you rename or move example folders, keep legacy content under `examples/legacy/` (no deletions without approval).
- Update docs alongside code changes (handbook/guide sections).

## Absolute NO List
- **Astrabot** (`astrabot.com`)
  - Reason: Conflict of interest / clean room violation.
  - Action: Reject commits, purge history.

