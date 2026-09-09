---
name: lint-skills
description: Audit this repo's skills and command stubs — the thin-skill contract, provenance, live script paths, a stub that doubles or restates its own skill, and absolute paths written out where a variable belongs. Use when asked whether the skills are still consistent, or after adding one.
license: MIT
compatibility: Requires git and `uv` or `python3`. Runs in a claude-meta clone or any bootstrapped project; the checkers ship as class-M `bin/**`, so a project that predates them is told to run `/alemax:complete-update` rather than failing.
context: either
argument-hint: "[--strict] [--include-specs] [--repo <path>]"
allowed-tools: Bash(uv run --script .claude/skills/alemax/skills/lint-all/scripts/claude_lint.py *) Bash(python3 .claude/skills/alemax/skills/lint-all/scripts/claude_lint.py *)
metadata:
  author: alemax
  version: "1.0"
  reviewed_model: claude-5
  reviewed: 2026-09-07
---

## Wraps

`alemax lint` — `all`, `docs`, `skills`. The shipped entry point is
`.claude/skills/alemax/skills/lint-all/scripts/claude_lint.py`; run it from the repo root with `uv run --script`,
or `python3` in its place. `--help` documents every flag.

Exit codes, branching and the reasoning behind each rule are in spec `claude-consistency-lint`
and the module's own docstrings. This body carries what a session must decide, and what
to report — restating the code here is how the two drift.

## Steps

1. **Just run it.** The runner finds the checkers (`bin/` in a project,
   `scaffolding/templates/bin/` in claude-meta). `--repo` only when the operator names another
   clone — a repo with its own session is reached by `/alemax:send-msg`, never audited from here.
2. **Read the verdict, not the volume.** Every finding is printed; `--strict` is what makes one
   FAIL, and is what to pass before a broadcast.
3. **Report by cause.** D1 is the listing budget (two entries, one capability — the fix is a
   plugin and a deleted stub). D2 is drift-room, not drift: a routing line cannot contradict its
   skill; a paraphrase can, and every alemax pair did. P1 is a path that names one machine.
4. **A P1 is a fix or a stated exception, never a silent one.** Where the literal is deliberate —
   `$HOME` redirected, so the boot-disk path IS the real one — the answer is a line in
   `.claude/path-literal-allow.txt` with a trailing `# <reason>`; the checker refuses an entry
   that has no reason. Where the line is the rule's own example or a test fixture, the answer is
   a same-line `path-literal-ok` / `path-literal-docs`. Ask the operator which; never quieten it.
5. **Fix nothing without asking.** A skill body is somebody's contract.

## Not for

- The documents and the change triad → `/alemax:lint-docs`; both at once → `/alemax:lint-all`.
- Authoring a new skill → `meta/docs/ALEMAX-SKILLS.md` § Adding a new `/alemax:*` skill.
- `$HOME/Library/…` inside a skill body — the opposite rule (`system-path-rule`), <!-- system-path-rule-docs -->
  enforced by `meta/scripts/hooks/system-path-rule-check.sh`. The two agree at the account name.
