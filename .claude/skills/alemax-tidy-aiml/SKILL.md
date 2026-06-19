---
name: alemax-tidy-aiml
description: Reshape the current aiml0NN per-drive branch into the canonical operator-state shape — origin/main + one tidy commit per yaml file (orgs, users, repos, projects) + any fork-divergent non-yaml commits preserved on top. Conversational front-end for meta/scripts/fork-tidy-aiml.sh. Use when the operator's aiml0NN history has drifted (multiple commits per yaml, mixed-file commits, merge-commit chains from fork-sync) and they want it back to the clean linear shape.
license: MIT
compatibility: Requires bash, git, and the alemaxdesign/claude-meta fork model live.
context: claude-meta-only
metadata:
  author: alemax
  version: "1.0"
---

## Wraps

`meta/scripts/fork-tidy-aiml.sh` — the reshape script. It does the actual classification, temp-branch build, tree-equivalence check, and force-push-with-lease. This skill never duplicates that logic; it surfaces the preflight, presents the plan conversationally, confirms, invokes, reports.

## When to use

Run when:

- A new operator session sees `git log origin/main..aiml0NN` with more commits than expected, or with `chore(manifest):` commits touching multiple yaml files in one go.
- After a sync round where merge commits accumulated and the operator wants a linear, reviewable shape.
- After a manual `git commit` on an aiml0NN that wasn't named in canonical `chore(<scope>): aiml0NN operator state` form.

Skip when:

- `git log --oneline origin/main..aiml0NN` already shows exactly the tidy shape (one commit per yaml file in order, then any non-yaml commits). The script is idempotent — running it is safe but uninformative — but the skill should still spot this and tell the operator "no work to do" without ceremony.
- The operator is on `main`, canonical, or a feature branch. The script refuses; the skill should refuse first with a clearer hint.

## Behaviour overview

1. **Detect context** — confirm we're in claude-meta on an aiml0NN branch.
2. **Preflight** — clean tree, origin/main not ahead (`fork-sync.sh` first if it is), no mixed-file commits.
3. **Classify + preview** — show the operator the current shape, the proposed shape, and the diff.
4. **Confirm** — wait for explicit "yes" before the rewrite (force-push is involved).
5. **Invoke** — `meta/scripts/fork-tidy-aiml.sh --yes [--push]` based on whether the operator wants the push in one step.
6. **Report** — show the new `git log origin/main..aiml0NN` and remind the operator of the cherry-pick consideration if relevant.

## Steps

### Step 0 — Context check

```bash
# This skill declares context: claude-meta-only.
ORIGIN_URL="$(git remote get-url origin 2>/dev/null || true)"
CURRENT_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ "$ORIGIN_URL" != *"claude-meta"* || "$CURRENT_ROOT" != *"/claude-meta" ]]; then
  echo "ERROR: alemax-tidy-aiml requires claude-meta-only context." >&2
  echo "  current: $CURRENT_ROOT" >&2
  echo "  origin:  $ORIGIN_URL" >&2
  echo "cd to your meta-repo fork clone first." >&2
  exit 1
fi

# Refuse on canonical.
if [[ "$ORIGIN_URL" == *"alemaxdesign/claude-meta"* ]]; then
  echo "ERROR: origin is canonical (alemaxdesign/claude-meta)." >&2
  echo "This skill rewrites fork-divergent aiml0NN branches; canonical has no such branches." >&2
  exit 1
fi

# Branch assertion.
CURRENT_BRANCH="$(git symbolic-ref --short HEAD 2>/dev/null || echo "<detached>")"
if [[ ! "$CURRENT_BRANCH" =~ ^aiml[0-9]{2}$ ]]; then
  echo "ERROR: current branch '$CURRENT_BRANCH' is not aiml0NN." >&2
  echo "Switch first: git checkout aiml0NN" >&2
  exit 1
fi
```

### Step 1 — Preflight checks

```bash
# Working tree clean
[[ -z "$(git status --porcelain)" ]] || { echo "tree dirty"; exit 1; }

# Local origin/main not ahead of $CURRENT_BRANCH (would mean fork-sync needed)
git fetch origin --quiet
BEHIND="$(git rev-list --count "$CURRENT_BRANCH..origin/main" 2>/dev/null || echo 0)"
if [[ "$BEHIND" -gt 0 ]]; then
  echo "origin/main is $BEHIND commit(s) ahead of $CURRENT_BRANCH." >&2
  echo "Run \`./meta/scripts/fork-sync.sh --non-interactive --push\` first, then retry." >&2
  exit 1
fi
```

If any preflight fails, **stop** with the specific fix command. Do not proceed.

### Step 2 — Classify + dry-run preview

Run the script in dry-run mode first to surface mixed-file commits and the proposed plan:

```bash
./meta/scripts/fork-tidy-aiml.sh --dry-run
```

Three outcomes:

| Script exit | Meaning | Skill response |
|---|---|---|
| 0, says "already tidy" | No reshape needed | Report "Nothing to tidy" and stop. |
| 0, prints plan | Reshape is safe | Proceed to Step 3. |
| non-zero, mixed-file commits | Operator must split first | Show the offending SHAs + the `git rebase -i <base>` hint. Stop. |
| non-zero, other | Investigate | Surface the script's stderr verbatim; stop. |

### Step 3 — Present the plan + confirm

Use the **AskUserQuestion tool**. The plan has two decisions stacked:

1. Apply the reshape locally?
2. Push the result in the same step, or leave the operator to push separately?

Default-recommended option: apply + push. Force-push-with-lease on an operator-owned `aiml0NN` is the canonical workflow and removes a manual follow-up. The operator can opt out if they want to eyeball `git log` before publishing.

### Step 4 — Invoke

```bash
# With push in one step:
./meta/scripts/fork-tidy-aiml.sh --yes --push

# Local only:
./meta/scripts/fork-tidy-aiml.sh --yes
```

The `--yes` skips the script's interactive confirmation (the skill already confirmed).

### Step 5 — Report

After success, run:

```bash
git log --oneline origin/main..HEAD
```

and surface the resulting shape to the operator. Three follow-ups to flag if relevant:

- **If non-yaml commits were preserved on top:** name them ("one tooling commit preserved: `<short-sha> <subject>`") so the operator can decide whether to open a canonical PR for any that belong upstream.
- **If the operator did `--yes` without `--push`:** remind them of the `git push --force-with-lease origin <branch>` follow-up.
- **If the script also tidied a recent in-flight session's commits:** mention which fork-divergent SHAs disappeared so the operator can confirm none were ones they wanted to keep.

## Failure modes

| Symptom | Cause | Recovery |
|---|---|---|
| Script refuses with "mixed yaml + non-yaml commit" | A single commit touches e.g. `orgs.yaml` AND `meta/scripts/foo.sh` | Run `git rebase -i origin/main`, split that commit into two (`git reset HEAD^` + two `git commit`s), then re-invoke. |
| Cherry-pick conflict on a non-yaml commit | A non-yaml commit on `aiml0NN` no longer applies cleanly atop the 4 tidy yaml commits (rare; usually means the yaml-only commits touched something the non-yaml commit depends on — they shouldn't, but corner case). | Script leaves the temp branch in a conflicted cherry-pick state. Operator resolves with `git status` + `git cherry-pick --continue`. |
| Tree-equivalence check fails | Logic bug in this script (shouldn't happen; if it does, the temp branch is preserved for diffing). | File a feedback note via `/alemax:feedback`; do NOT force the reshape. |
| Force-push rejected (`--force-with-lease` mismatch) | Someone else's session updated `origin/aiml0NN` between fetch and push. | Re-run `fork-sync.sh` to absorb the remote state, then re-run `tidy-aiml`. |

## Why this skill exists

`fork-sync.sh` is **merge-only** by design — it's the safe, additive path that never rewrites history. Reshape is fundamentally **destructive** (force-push), so it lives in its own tool + skill with an explicit confirmation. Mixing the two into one script would either over-empower the safe path or under-power the destructive one. See `meta/docs/HISTORY-AND-SQUASH-POLICY.md` (when shipped) for the policy context; the `aiml0NN` rebase cadence noted there is what this skill operationalizes.
