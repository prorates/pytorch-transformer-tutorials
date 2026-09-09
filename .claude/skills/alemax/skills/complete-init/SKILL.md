---
name: complete-init
description: Project-side completion of a freshly-bootstrapped project — run from the NEW project's own Claude session after `/alemax:new-project` created and pushed it. `scripts/init.py` refuses on the meta-repo, gates on freshness (complete-init finishes a fresh bootstrap; on a long-running clone its writes clobber choices the project now owns), lists every gap the bootstrap left, and applies the two safe ones. This session then does the rest: Keychain secrets, `origin/HEAD`, the settings reconcile, the ci.yml trim, the go-stack post-generate, verification, and the first real change through to a merged PR. Consumer half of `/alemax:new-project`.
license: MIT
compatibility: Requires git, gh, and `uv` or `python3`. Runs in the project clone; refuses on a claude-meta clone and on a project that is no longer fresh (`--force` overrides).
context: project
argument-hint: "[--dry-run] [--force]"
allowed-tools: Bash(uv run --script .claude/skills/alemax/skills/complete-init/scripts/init.py *) Bash(python3 .claude/skills/alemax/skills/complete-init/scripts/init.py *)
metadata:
  author: alemax
  version: "3.0"
  reviewed_model: claude-5
  reviewed: 2026-09-07
---

## Wraps

`alemax init` — `context`, `fix`, `gaps`. The shipped entry point is
`.claude/skills/alemax/skills/complete-init/scripts/init.py`; run it from the repo root with `uv run --script`,
or `python3` in its place. `--help` documents every flag.

Exit codes, branching and the reasoning behind each rule are in spec `project-side-completion`
and the module's own docstrings. This body carries what a session must decide, and what
to report — restating the code here is how the two drift.

## Steps

1. **`context`, then `gaps`.** `context` refusing means the meta-repo (use `/alemax:new-project`
   there), no `.meta-version` (retrofitting is meta-side), or a clone past a fresh bootstrap.
   **That last one is the real guard** — these writes are safe only on a fresh clone. Pass
   `--force` only when the operator says they mean it, and say what it will overwrite first.
2. **`fix --gap gitignore-settings`, then `fix --gap python-version`.** In that order:
   `.claude/settings.local.json` holds the operator's grants and must be ignored **before**
   anything writes to it.
3. **Reconcile the settings floor** — `bin/reconcile-settings.py check`, then `apply`. Before the
   first PR, so step 7's `gh pr merge` does not hit a permission wall mid-flow. If the harness
   refuses the write, `apply --stage` prints a `cp` line to hand the operator.
4. **Close the advisory gaps** `gaps` listed: `git remote set-head origin --auto`,
   `pre-commit install`, the Keychain secrets under the service name `context` reports, and
   mirroring any CI needs with `bin/sync-secrets.py` (names only — values are never printed).
   Delete the `ci.yml` jobs for a stack this repo lacks: runtime gating stops them *running*, not
   Dependabot, which parses the workflow statically and opens bump PRs anyway.
5. **Go stack only:** `make manifests generate`, then guide the rename of the sample API. Domain
   modelling is the operator's — do not invent the resource.
6. **Verify.** Run the project's own tests and report pass or fail without auto-fixing. Propose,
   do not run, `/doctor`, `/test` and `/security`. A blocked run on a billing-capped org is not a
   failure.
7. **First real change, through to merged.** The operator decides what it is. Commit, push, open
   the PR, merge, confirm CI. This is where the end-to-end flow is actually exercised.
8. **Then drop the nudge** — remove the `CLAUDE.md` bootstrap block, but only once a real change
   has merged. If nothing has landed, leave it: it is supposed to keep prompting.
9. **Summarise** — done, skipped or pending, with the exact command for anything outstanding.

## Not for

The meta-repo, or a repo with no `.meta-version` — both refused at step 1. A long-running clone:
the freshness gate is the point. Removing the nudge before a first change merges. Applying a meta
*delivery* — that is `/alemax:complete-update`.
