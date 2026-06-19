---
name: "ALEMAX: Reprioritise ideas"
description: Curate § Suggested next-up of openspec/ideas.md — walk § Raw ideas, present `[ ]` entries with one-line summaries, let the operator pick 3–5 to surface as "next up", write pointer-only entries. Operator-triggered, often before planning sessions. Does NOT touch § Archived ideas or § Raw ideas content.
category: Workflow
tags: [workflow, openspec, alemax, planning]
---

<!-- Governance preamble — see openspec-governance-canonical-only (archived 2026-05-29) -->

## Governance preamble (run BEFORE any other step)

This command mutates `openspec/ideas.md`. The mutation lands via a feature branch + PR, never as a direct commit to fork main. The skill handles the branching internally.

Before doing anything else:

1. Run `git remote get-url origin` and `git branch --show-current` in the operator's meta-repo clone.
2. Apply the decision matrix:
   - **origin contains `alemaxdesign/claude-meta` AND branch = `main`** → proceed with a soft warning; the skill will create a feature branch internally.
   - **origin is a fork AND branch = `main`** → proceed; the skill will branch off `main` internally.
   - **origin is a fork AND branch ≠ `main` (feature branch)** → proceed; the skill will branch off `main` internally.
3. Verify the working tree is clean in the meta-repo clone. The skill refuses to proceed on a dirty tree.

Once preflight passes, **delegate the rest to the `alemax-reprioritise-ideas` skill** (`.claude/skills/alemax-reprioritise-ideas/SKILL.md`).

---

## Context guard

This command requires the operator to be in their **claude-meta clone** (meta-repo), not a downstream project. The skill body's Step 1 Preflight verifies this and refuses on mismatch. If you're seeing this skill listed from a project clone session (post-`ship-alemax-skills-in-projects`), `cd` to your meta-repo clone first.

Skill declared context: `claude-meta-only` (per `.claude/skills/alemax-reprioritise-ideas/SKILL.md` frontmatter, codified in `openspec/specs/alemax-skills/spec.md`).

---

## Input

The argument after `/alemax:reprioritise-ideas` is optional:

- **No args** — default behavior: present § Raw ideas list, conversational pick.
- **`--clear`** — clear § Suggested next-up (no new entries, replace with empty). Useful after a planning session whose picks have all shipped.

Examples:

```
/alemax:reprioritise-ideas
/alemax:reprioritise-ideas --clear
```

## Steps

1. **Preflight** — verify claude-meta clone, clean working tree.
2. **Walk § Raw ideas** — extract every `[ ]` entry's slug + one-line summary.
3. **Present + pick** — show the numbered list; operator picks 3–5 by index or slug.
4. **Detect existing § Suggested next-up** — if non-empty, ask Replace / Add to / Skip.
5. **Atomic write** the mutated `openspec/ideas.md`.
6. **Branch + commit + push + PR** to canonical.
7. **Summary**.

Full procedure in `.claude/skills/alemax-reprioritise-ideas/SKILL.md`.

## When NOT to use this skill

- **§ Raw ideas has zero `[ ]` entries** — nothing to surface; skill exits cleanly.
- **You want to ADD just one entry to § Suggested next-up** — that's fine; the skill supports the "add" mode when existing entries are present. But hand-editing the file is also OK for a single-entry add.
- **§ Suggested next-up is meant to mirror § Raw ideas** — it's not. § Suggested next-up is for a curated 3–5; if you want the full list, just scroll to § Raw ideas.

## Cross-links

- `.claude/skills/alemax-reprioritise-ideas/SKILL.md` — full skill body.
- `openspec/specs/spec-governance/spec.md` — the rhythm-based ideas.md structure this skill operates on.
- `openspec/specs/alemax-skills/spec.md` — family conventions.
- `/alemax:archive-ideas` — sibling skill for § Archived ideas — by capability reshape.
