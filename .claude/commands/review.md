---
description: Review the current diff (uncommitted or vs main). Adversarial code review.
---

# /review

Run a PR-style review on the current diff. Default scope: uncommitted changes (`git diff` + `git diff --staged`). If the working tree is clean, fall back to the diff between current branch and `main`.

## Procedure

1. Determine scope.
   - If there are uncommitted changes: review them.
   - Otherwise: `git diff main...HEAD`.
2. Read the changed files in their entirety, not just the hunks. Hunks lie about context.
3. Delegate to the `code-reviewer` subagent for the actual review pass.
4. Cluster findings by severity:
   - **🚨 Blockers** — correctness bugs, security issues, broken tests.
   - **⚠️ Concerns** — design smells, missing edge cases, weak tests.
   - **💡 Nits** — style, naming, micro-optimizations.
5. Quote file paths as `path/to/file.py:42` so the user can click straight to the line.

## What to look for

- Correctness: does the change do what the diff/commit message claims?
- Edge cases: empty input, large input, unicode, None/null, concurrent calls.
- Test coverage: are the new code paths exercised? Negative cases too?
- Security: secrets, injection, unsafe deserialization, missing input validation at boundaries.
- Style/readability: misleading names, dead code, layered abstractions hiding intent.
- Docs: do the README/CLAUDE.md still match reality?

## Don't

- Don't auto-apply fixes from a review unless the user asks.
- Don't pad the review with praise. Note what's good only if it's worth modeling.
