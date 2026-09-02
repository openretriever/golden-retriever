# Repository Guidance

Follow [`AGENTS.md`](AGENTS.md) as the canonical repository guidance.

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
