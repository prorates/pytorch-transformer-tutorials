---
name: "ALEMAX: Archive ideas"
description: Periodic reshape of openspec/ideas.md — for each `[x]` entry in § Raw ideas, classify by capability via the corresponding archived change's proposal, snapshot the file, condense + move into § Archived ideas — by capability. Operator-triggered, batched, per-entry confirmation.
category: Workflow
tags: [workflow, openspec, alemax, reshape]
---

<!-- Governance preamble — see openspec-governance-canonical-only (archived 2026-05-29) -->

## Governance preamble (run BEFORE any other step)

This command mutates `openspec/ideas.md` and adds files under `openspec/ideas-snapshots/`. Both are canonical-only — mutations land via a feature branch + PR, never as direct commits to fork main. The skill handles the branching internally.

Before doing anything else:

1. Run `git remote get-url origin` and `git branch --show-current` in the operator's meta-repo clone.
2. Apply the decision matrix:
   - **origin contains `alemaxdesign/claude-meta` AND branch = `main`** → proceed with a soft warning; the skill will create a feature branch internally.
   - **origin is a fork AND branch = `main`** → proceed; the skill will branch off `main` internally.
   - **origin is a fork AND branch ≠ `main` (feature branch)** → proceed; the skill will branch off `main` internally (the reshape PR goes to canonical, not the current feature branch).
3. Verify the working tree is clean in the meta-repo clone. The skill refuses to proceed on a dirty tree.

Once preflight passes, **delegate the rest to the `alemax-archive-ideas` skill** (`.claude/skills/alemax-archive-ideas/SKILL.md`).

---

## Context guard

This command requires the operator to be in their **claude-meta clone** (meta-repo), not a downstream project. The skill body's Step 1 Preflight verifies this and refuses on mismatch. If you're seeing this skill listed from a project clone session (post-`ship-alemax-skills-in-projects`), `cd` to your meta-repo clone first.

Skill declared context: `claude-meta-only` (per `.claude/skills/alemax-archive-ideas/SKILL.md` frontmatter, codified in `openspec/specs/alemax-skills/spec.md`).

---

## Input

The argument after `/alemax:archive-ideas` is optional:

- **No args** — default behavior: snapshot to `openspec/ideas-snapshots/YYYY-MM-DD-pre-reshape.md`, per-entry confirmation, commit + PR.
- **`--context <name>`** — use a custom snapshot suffix: `openspec/ideas-snapshots/YYYY-MM-DD-pre-<name>.md`. Useful for ad-hoc reshapes with specific motivation (e.g., `--context pre-american-dream-demo`).
- **`--yes-all`** — skip per-entry confirmation; auto-accept the tier-1/tier-2 classification result for every entry. Tier-3 (operator prompt) still fires for ambiguous entries. Use only after eyeballing § Raw ideas.

Examples:

```
/alemax:archive-ideas
/alemax:archive-ideas --context pre-demo
/alemax:archive-ideas --yes-all
```

## Steps

1. **Preflight** — verify claude-meta clone, clean working tree.
2. **Snapshot** — copy `openspec/ideas.md` to the dated snapshot path.
3. **Enumerate `[x]` entries** in § Raw ideas.
4. **Classify each** via 3-tier fallback (inline pointer → body slug-mention → operator prompt).
5. **Confirm each** (`y` / `skip` / `edit`) unless `--yes-all`.
6. **Atomic write** the mutated `openspec/ideas.md`.
7. **Branch + commit + push + PR** to canonical.
8. **Summary**.

Full procedure in `.claude/skills/alemax-archive-ideas/SKILL.md`.

## When NOT to use this skill

- **Right after a single archive** — wait for 3+ `[x]` entries to accumulate. One-off reshapes have low signal-to-noise ratio.
- **Inside a fork feature branch you intend to keep separate** — the reshape always targets canonical via PR; if you're on a feature branch, the reshape branches off `main`, not your current branch.
- **When § Raw ideas has zero `[x]` entries** — skill exits cleanly with "Nothing to reshape" but the operator gets no value from the run.

## Cross-links

- `.claude/skills/alemax-archive-ideas/SKILL.md` — full skill body.
- `openspec/specs/spec-governance/spec.md` — the lifecycle this skill implements.
- `openspec/specs/alemax-skills/spec.md` — family conventions (system-path-rule, conversational pattern).
- `/alemax:reprioritise-ideas` — sibling skill for § Suggested next-up curation.
