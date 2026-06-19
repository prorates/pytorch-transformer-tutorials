---
name: alemax-archive-ideas
description: Periodic reshape of openspec/ideas.md — for each `[x]` entry in § Raw ideas, classify by reading the corresponding archived change's `proposal.md § Capabilities`, snapshot `openspec/ideas.md` to `openspec/ideas-snapshots/YYYY-MM-DD-pre-reshape.md`, append a one-line bullet under each matching capability heading in § Archived ideas — by capability, and remove the full body from § Raw ideas. Operator-triggered + per-entry confirmation. Run when § Raw ideas has accumulated `[x]` entries (weekly, after N archives, before a demo).
license: MIT
compatibility: Requires bash, git, gh. Operates exclusively under `openspec/` (no macOS-system-rooted paths). Honors the canonical-only governance rule per [[openspec-governance-canonical-only]] (archived 2026-05-29).
context: claude-meta-only
metadata:
  author: alemax
  version: "1.0"
---

## Wraps

No single shell script. The workflow is: walk `openspec/ideas.md` § Raw ideas, find `[x]` entries, classify each by capability via the archived change's proposal, condense + move into § Archived ideas — by capability.

The skill embodies the lifecycle codified in `openspec/specs/spec-governance/spec.md` (the "rhythm-based 5-section structure with a dual-trigger lifecycle" requirement). `/opsx:archive` does the one-character `[ ]` → `[x]` flip; this skill does the periodic structural reshape that batches the cleanup.

## The reshape story

`openspec/ideas.md` is structured into:

1. `## How to walk this file in an autonomous-mode session` (stable)
2. `## Reshape + snapshot convention` (stable)
3. `## Archived ideas — by capability` (rewritten only by this skill)
4. `## Suggested next-up` (rewritten by `/alemax:reprioritise-ideas`)
5. `## Raw ideas` (append-mostly; `[x]` accumulates between reshapes)

Between reshapes, `[x]` entries accumulate in § Raw ideas — `/opsx:archive` flips the box but does NOT touch section structure. This skill is the operator-triggered batch cleanup.

## Behavior overview

1. **Preflight.** Resolve the meta-repo root (`git rev-parse --show-toplevel`); verify it's claude-meta (origin URL contains `claude-meta`); verify working tree is clean.
2. **Snapshot.** Copy current `openspec/ideas.md` to `openspec/ideas-snapshots/YYYY-MM-DD-pre-reshape.md`. Handle filename collisions (append `-2`, `-3`, …). Operator may pass `--context <name>` for a custom suffix.
3. **Walk § Raw ideas.** Extract every `[x]` entry (bullet + indented continuation lines).
4. **Per-entry classification (3-tier fallback).** For each `[x]` entry:
   - **Tier 1 — inline pointer**: match `→ \[(?P<slug>[a-z0-9-]+)\]\(changes/archive/[^)]+\)` in the entry body. If found, use `<slug>` and read `openspec/changes/archive/YYYY-MM-DD-<slug>/proposal.md` for the `## Capabilities` section.
   - **Tier 2 — body slug-mention**: if no inline pointer, grep the entry body for backtick-wrapped slug-like tokens matching any name in `openspec/changes/archive/`. If exactly one match, use it. If multiple or zero, fall through.
   - **Tier 3 — operator prompt**: present the entry's first non-empty line + the list of recent archive slugs, ask "Which archived change does this map to? [slug or skip]".
5. **Per-entry confirmation.** Once classified, present `Reshape under <capability>(s)? [y/skip/edit]`:
   - `y` (default): append the one-line bullet under each named capability heading; mark entry for removal from § Raw ideas.
   - `skip`: leave the `[x]` entry in place; it'll be picked up at the next reshape.
   - `edit`: present the full list of capability headings under § Archived ideas; operator picks one or more; proceed with the new selection.
6. **Atomic write.** After all entries processed, write the mutated `openspec/ideas.md` via tempfile + `mv`.
7. **Branch + commit + PR.** Create a `chore/reshape-ideas-<date>` branch off `main`, commit with a message summarizing what was reshaped, push, open a PR via `gh pr create`.
8. **Final summary.** Report counts: N entries classified, M reshaped, K skipped, snapshot path, PR URL.

## Steps

### Step 1 — Preflight (Context check + clean tree)

```bash
# Context check — this skill declares context: claude-meta-only.
# Refuse if invoked from outside the meta-repo (per alemax-skills/spec.md).
CURRENT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
ORIGIN_URL="$(git remote get-url origin 2>/dev/null || true)"
if [[ "$ORIGIN_URL" == *"claude-meta"* && "$CURRENT_ROOT" == *"/claude-meta" ]]; then
  CURRENT_CONTEXT="claude-meta"
else
  CURRENT_CONTEXT="project"
fi
if [[ "$CURRENT_CONTEXT" != "claude-meta" ]]; then
  echo "ERROR: alemax-archive-ideas requires context: claude-meta-only but you're in: $CURRENT_CONTEXT" >&2
  echo "  current:    $CURRENT_ROOT" >&2
  echo "" >&2
  echo "To proceed, cd to your meta-repo clone (typically: /Volumes/AIML0NN/Users/<u>/claude-code/<ghhandle>/claude-meta)." >&2
  exit 1
fi
META_ROOT="$CURRENT_ROOT"

# Clean tree check
if ! git diff-index --quiet HEAD --; then
  echo "ERROR: working tree dirty — commit or stash first" >&2
  exit 1
fi
```

### Step 2 — Snapshot

```bash
SNAPSHOT_DIR="$META_ROOT/openspec/ideas-snapshots"
TODAY="$(date +%Y-%m-%d)"
CONTEXT="${CONTEXT:-pre-reshape}"
SNAPSHOT="$SNAPSHOT_DIR/$TODAY-$CONTEXT.md"
N=2
while [ -e "$SNAPSHOT" ]; do
  SNAPSHOT="$SNAPSHOT_DIR/$TODAY-$CONTEXT-$N.md"
  N=$((N+1))
done
cp "$META_ROOT/openspec/ideas.md" "$SNAPSHOT"
echo "Snapshot: $SNAPSHOT"
```

### Step 3 — Walk § Raw ideas

```bash
awk '
  /^## Raw ideas$/ { in_raw=1; next }
  /^## / { in_raw=0 }
  in_raw && /^- \[x\]/ { print NR ":" $0; entry_open=1; next }
  in_raw && entry_open && /^- \[/ { entry_open=0 }
  in_raw && entry_open { print NR ":" $0 }
' openspec/ideas.md
```

Yields a stream of `<line-number>:<content>` records grouped by `[x]` entry.

### Step 4 — Per-entry classification

For each `[x]` entry body:

**Tier 1** — extract inline pointer:
```bash
SLUG=$(grep -oE '→ \[[a-z0-9-]+\]\(changes/archive/[^)]+\)' <<<"$ENTRY_BODY" | sed -E 's/.*\[([^]]+)\].*/\1/')
```
If `$SLUG` non-empty AND `openspec/changes/archive/<YYYY-MM-DD>-$SLUG/proposal.md` exists → read its § Capabilities.

**Tier 2** — body slug-mention scan:
```bash
ARCHIVE_SLUGS=$(ls openspec/changes/archive/ | sed -E 's/^[0-9-]+-//' | sort -u)
MATCHES=$(echo "$ARCHIVE_SLUGS" | while read S; do grep -qE "\b$S\b" <<<"$ENTRY_BODY" && echo "$S"; done)
COUNT=$(echo "$MATCHES" | wc -l)
[ "$COUNT" = "1" ] && SLUG="$MATCHES"
```

**Tier 3** — operator prompt:
```
The following entry has no clear archive pointer:

  > [first line of entry body]

Recent archive slugs:
  - retrofit-existing-repos
  - multi-user-environment-bootstrap
  - ...

Which archived change does this map to? [slug or skip]: ___
```

### Step 5 — Per-entry confirmation

Read target capability heading(s) from the archived change's `proposal.md § Capabilities` section. Look for `### Modified Capabilities` and `### New Capabilities` subsections; extract capability names (the `<name>` from `- **\`<name>\`**:` patterns).

Present:
```
Reshape entry "<first line>" under <capability>? [y/skip/edit]: ___
```

### Step 6 — Atomic write

Build the mutated `openspec/ideas.md` in a tempfile:
- Insert each one-line bullet `- YYYY-MM-DD — <slug> ([archive](changes/archive/...)) — <one-line summary>` under the matching `### [` capability heading in § Archived ideas.
- Remove the corresponding `[x]` entry (bullet + indented continuation) from § Raw ideas.

```bash
TMPFILE="$(mktemp)"
# build mutated content in $TMPFILE
mv "$TMPFILE" openspec/ideas.md
```

### Step 7 — Branch + commit + PR

```bash
BRANCH="chore/reshape-ideas-$TODAY"
git checkout -b "$BRANCH"
git add openspec/ideas.md openspec/ideas-snapshots/
git commit -m "chore(openspec): reshape ideas.md — N entries to § Archived"
git push -u origin "$BRANCH"
gh pr create --base main --title "chore(openspec): reshape ideas.md ($TODAY)" --body "..."
```

### Step 8 — Final summary

```
Reshape complete.
  Snapshot:    openspec/ideas-snapshots/YYYY-MM-DD-pre-reshape.md
  Classified:  N entries
  Reshaped:    M
  Skipped:     K
  PR:          https://github.com/alemaxdesign/claude-meta/pull/<num>
```

## Edge cases

- **No `[x]` entries** → skill exits with "No `[x]` entries to reshape" and no mutation.
- **Multi-capability changes** → bullet appended under each named capability heading. Operator informed but not re-prompted per capability.
- **Operator chooses `edit` on every entry** → equivalent to manual classification; skill proceeds entry-by-entry as the operator directs.
- **Snapshot filename collision** → append `-2`, `-3`, …. Logged so operator can find the snapshot.
- **Archive proposal.md missing `## Capabilities` section** → fall through to tier-3 operator prompt with a warning.

## system-path-rule

This skill operates exclusively under `openspec/` and does NOT reference macOS-system-rooted paths. If future enhancements need a `~/Library/...` path, use literal `/Users/$(whoami)/...` per `openspec/specs/alemax-skills/spec.md` system-path-rule requirement. <!-- system-path-rule-docs -->

## Cross-links

- `openspec/specs/spec-governance/spec.md` — the rhythm-based ideas.md structure + dual-trigger lifecycle requirement this skill implements.
- `openspec/specs/alemax-skills/spec.md` — family conventions (system-path-rule, conversational pattern, governance preamble).
- `openspec/changes/archive/2026-06-01-spec-summaries-and-ideas-archive/` — the change that created the structure this skill operates on.
- `.claude/skills/alemax-reprioritise-ideas/SKILL.md` — sibling skill for § Suggested next-up curation.
