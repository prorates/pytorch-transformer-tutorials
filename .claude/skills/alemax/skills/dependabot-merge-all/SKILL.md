---
name: dependabot-merge-all
description: Merge several Dependabot PRs in sequence, re-rebasing each after the previous one lands — squash-merging one flips every other PR touching the same file to DIRTY (their diff used the now-changed line as context), so each needs its own `@dependabot rebase` round-trip. Same script as `/alemax:dependabot-merge`, run with `--all` or an ordered list; stops at the first PR that stops. Works from any repo with a GitHub remote.
license: MIT
compatibility: Requires `gh` (authenticated), git, and `uv` or `python3`. Merges only with `--apply`.
context: either
argument-hint: "--all | <pr> <pr> … [--apply]"
allowed-tools: Bash(uv run --script .claude/skills/alemax/skills/dependabot-merge/scripts/dependabot_merge.py *) Bash(python3 .claude/skills/alemax/skills/dependabot-merge/scripts/dependabot_merge.py *)
metadata:
  author: alemax
  version: "3.0"
  reviewed_model: claude-5
  reviewed: 2026-09-07
---

## Wraps

`alemax dependabot` — no subcommands — flags only. The shipped entry point is
`.claude/skills/alemax/skills/dependabot-merge/scripts/dependabot_merge.py`; run it from the repo root with `uv run --script`,
or `python3` in its place. `--help` documents every flag.

Exit codes, branching and the reasoning behind each rule are in spec `alemax-dependabot-skills`
and the module's own docstrings. This body carries what a session must decide, and what
to report — restating the code here is how the two drift.

## Steps

1. **Dry run:** `uv run --script .claude/skills/alemax/skills/dependabot-merge/scripts/dependabot_merge.py --all`
   (or the PR numbers in the order wanted) — confirm the list with the operator.
2. **Apply:** the same line with `--apply`. Each subsequent PR is rebased by the script after the
   previous merge; do not comment or poll by hand.
3. **Report** the summary block — which merged (sha), which stopped and why, which are untouched.
   On a stop the operator fixes the blocker and re-runs with the remaining numbers.

## Not for

- One PR → `/alemax:dependabot-merge <pr>`.
- PRs touching different files — they merge independently; batching adds nothing.
- Bumps the operator has not accepted — decide first, then batch.
