# CLAUDE.md

This file orients Claude Code at the start of every session in this repo.

> **Just bootstrapped via `/alemax:new-project`?** Run `/opsx:propose` to spec out your first change.
> *(This nudge can be removed once you've made your first commit beyond bootstrap.)*

## Project purpose

<TBD>

## Identity

- **GitHub owner:** `prorates` (where `gh` pushes; where the repo lives)
- **Commits authored as:** `{{OPERATOR_NAME}} <{{OPERATOR_EMAIL}}>` (asserted at the repo level by `init-project.sh`; matches `git config user.{name,email}`)
- **Keychain service:** `{{KEYCHAIN_SERVICE}}` (where `bin/load-secrets.sh` reads project secrets)

## Stack & layout

- **Stack:** bash
- **Owner:** `prorates` (GitHub)
- **Bootstrapped from:** [`claude-meta@f04f6d86ed7308cb88958836bc1c16dd8538d75c`](https://github.com/alemaxdesign/claude-meta) on 2026-06-19T21:17:04Z

### Python projects
- Source under `src/<package>/`, tests under `tests/`.
- Package management: `uv`. Lockfile: `uv.lock`. Run things via `uv run …`.
- Type-checked with `mypy --strict`.
- Linted/formatted with `ruff`.

### Bash projects
- Entrypoints under `bin/`, helpers under `lib/`.
- Tests with `bats` under `tests/`.
- Linted with `shellcheck`, formatted with `shfmt`.

## Available commands

Slash commands live in `.claude/commands/`:

- `/plan` — Break down a feature into concrete steps.
- `/review` — Review the current diff (uncommitted or vs `main`).
- `/test` — Run tests; if failing, propose fixes; if missing, generate.
- `/debug` — Triage a failure (test output, stack trace, log).
- `/docs` — Update README and inline docs to match current state.
- `/security` — Scan for security issues (deps, secrets, common pitfalls).

## Available subagents

Subagents live in `.claude/agents/`:

- `code-reviewer` — Adversarial PR-style review.
- `test-writer` — Generates pytest/bats tests covering happy + sad paths + edges.
- `docs-writer` — README, CLAUDE.md, inline docstrings. Concise, example-driven.
- `security-scanner` — OWASP-style review, secret leaks, dep CVEs, unsafe defaults.

## Spec-driven development (OpenSpec)

This project ships with [OpenSpec](https://github.com/Fission-AI/OpenSpec) preinstalled. Use it for non-trivial changes — agree on what you're building before any code.

- `/opsx:propose "<idea>"` — Create a change with proposal, design, specs, and tasks.
- `/opsx:explore` — Brainstorm options before committing to one.
- `/opsx:apply` — Implement step-by-step against the task checklist.
- `/opsx:archive` — Archive the change once shipped; updates the main specs.

Specs live in `openspec/specs/`; in-flight changes live in `openspec/changes/`. The CLI works alongside the slash commands: `openspec list`, `openspec view`, `openspec validate`, `openspec status`.

For tiny edits (typo fix, single-line change), skip OpenSpec — overhead isn't worth it. For anything you'd want to think about for more than five minutes before coding, propose first.

## Development workflow

- Branches: `feat/<short>`, `fix/<short>`, `chore/<short>`.
- Commits: Conventional Commits (`type(scope): description`).
- All work merges via PR — even solo, to force CI to run.
- Squash-merge by default.
- PRs reference issues they close (`Closes #N`).

## Secrets

Local: macOS Keychain, service `{{KEYCHAIN_SERVICE}}`.
- Required keys are listed in `.env.example`.
- Populate: `bin/set-secret.sh <KEY>` (one key) or `bin/set-secret.sh --bootstrap` (walks every key in `.env.example`).
- Load into the current shell: `source bin/load-secrets.sh`.

CI: GitHub Actions repo secrets. Sync from Keychain with the meta-repo helper:
`<claude-meta>/meta/bin/sync-secrets-to-gh.sh --project pytorch-transformer-tutorials --ghhandle prorates`.

## Known gotchas

(Populated as the project matures.)

## Open questions

(Running list of unresolved design choices.)
