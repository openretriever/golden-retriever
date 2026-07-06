# Design notes — GoldenRetriever docs site

How this site should look and read. Read before touching
`src/styles/golden.css` or a page's structure.

## North star

Same discipline as the best developer docs — **Claude Code Docs**
(code.claude.com) and **OpenAI Developers** (developers.openai.com): calm,
generously spaced, quiet active states, tabbed code, restrained accent — with
the added warmth of **Claude/Anthropic**'s sites.

The one deliberate difference from core Retriever docs is the **ground**:

- **Core = white, precise, reference.**
- **Golden = warm "golden paper", applied, exploratory.**

Both share the same structure, type discipline, and restraint. Only the
background tone differs. That contrast is intentional — don't make Golden white,
and don't make core cream.

## What the references teach (and we follow)

1. **Restraint.** Mostly ink-on-ground with lots of air. The accent (orange)
   appears in small doses — links, one active state, a small highlight — not
   loud bars, gradients, or big fills.
2. **Quiet active states.** Active sidebar item = a soft pill, not a bold bar.
3. **Cohesive warm shell.** Page, sidebar, and header share one warm tone
   (`--sl-color-bg` / `--sl-color-bg-nav`); cards/panels lift on
   `--gold-surface`. Never a stark-white sidebar against a cream page.
4. **Generous whitespace, calm type scale**, prose measure ~65–72ch.
5. **Tabbed code / numbered steps** for multi-surface flows.
6. **One primary action per view.**

## Writing (Pixi discipline + no slop)

Command-first, lead with expected output. Concise, human prose — **avoid
AI-slop cadence** (parallel "X gives A, B, C; Y adds D, E, F" constructions,
empty tricolons, sentences that restate the heading). Honest positioning. Keep
expected-output blocks and the agent-first layer.

## Tokens (source of truth: `src/styles/golden.css`)

| Token | Light | Dark |
| --- | --- | --- |
| `--sl-color-bg` (page + shell) | `#f7f3ea` cream | `#1a1712` warm charcoal |
| `--sl-color-bg-nav` (sidebar/header) | `#f2ece0` (matches page) | `#1f1b15` |
| `--gold-surface` (cards/panels lift) | `#fffdf9` | `#262019` |
| `--sl-color-accent` | `#f97316` | `#fb923c` |
| `--sl-color-text-accent` | `#c2410c` | `#fdba74` |

Check every change in light **and** dark.

## Conventions

- **Eyebrow:** monospace, uppercase, wide tracking, 2px accent rule.
- **Cards:** `--gold-surface` fill, hairline border + subtle accent top-edge,
  gentle hover lift. Keep shadows soft.
- **Tables:** monospace headers, comfortable padding, quiet accent row-hover;
  spotlight the Retriever column in comparison matrices.

## Guardrails

Keep the warm ground cohesive (shell shares one tone). One accent hue. No
webfonts. No heavy shadows/gradients. `golden.css` is the only home for
site-wide tokens.

Companion: core Retriever's `docs-site/DESIGN.md` (clean white variant, same
discipline).
