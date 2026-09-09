"""init.py — the deterministic half of /alemax:complete-init.

Runs in the freshly-bootstrapped PROJECT's own clone. It detects what the
bootstrap left unfinished and applies only the safe project-local writes; it
never commits, pushes or opens a PR, and it never decides anything the
operator owns.

Subcommands
  context    project (not the meta-repo), `.meta-version` present, and the
             FRESHNESS gate — complete-init finishes a fresh bootstrap, and on
             a long-running clone its writes would clobber choices the project
             now owns. `--force` bypasses the gate and nothing else.
             Exit 0 fresh, 1 refused.
  gaps       every gap, each with a status and how it is closed.
             Exit 0 none open, 3 gaps open.
  fix        apply one safe gap: `python-version` · `gitignore-settings`.
             Everything else is advisory and stays the session's to run.
             Exit 0 applied, 1 refused.

The freshness gate is the primary guard against a mis-invocation; the per-step
confirmations in the body are the second.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from .. import git as gitmod
from ..io import die

FRESH_MAX_DEFAULT = 10
REQUIRES_RE = re.compile(r'requires-python\s*=\s*"[^"]*?(\d+\.\d+)')
JOB_RE = re.compile(r"^  (?P<job>[A-Za-z0-9_-]+):\s*$")
SETTINGS_IGNORE_RE = re.compile(r"(^|/)\.claude/settings\.local\.json\s*$")
NUDGE_RE = re.compile(
    r"Just bootstrapped|delete this note after your first real commit", re.IGNORECASE
)
STACK_MARKERS = {"python": "pyproject.toml", "go": "go.mod"}
STACK_HINTS = {
    "go": re.compile(r"setup-go|golangci|go\.mod", re.IGNORECASE),
    "python": re.compile(r"setup-uv|setup-python|pyproject|pytest", re.IGNORECASE),
}


def project_root() -> Path:
    top = gitmod.probe(Path.cwd(), "rev-parse", "--show-toplevel")
    if not top:
        die("not inside a git repository")
    return Path(top)


def detect_stack(root: Path) -> str:
    for stack, marker in STACK_MARKERS.items():
        if (root / marker).is_file():
            return stack
    if (root / "bin").is_dir() or list((root / "tests").glob("*.bats")):
        return "bash"
    return "unknown"


def context(root: Path, force: bool) -> dict:
    origin = (
        gitmod.probe(root, "remote", "get-url", "origin") or ""
    )  # no remote yet is a state, not a crash
    handle = ""
    if m := re.search(r"[:/]([^/]+)/[^/]+?(?:\.git)?$", origin):
        handle = m[1]
    commits = int(gitmod.probe(root, "rev-list", "--count", "HEAD") or 0)
    fresh_max = int(os.environ.get("COMPLETE_INIT_FRESH_MAX", FRESH_MAX_DEFAULT))
    out = {
        "root": str(root),
        "project": root.name,
        "origin": origin,
        "ghhandle": handle,
        "service": f"com.{handle}.{root.name}" if handle else None,
        "stack": detect_stack(root),
        "commits": commits,
        "fresh_max": fresh_max,
        "forced": force,
        "blocked": [],
    }
    if "claude-meta" in origin and root.name == "claude-meta":
        out["blocked"].append(
            "this is the claude-meta meta-repo — complete-init is context: project. "
            "To bootstrap a NEW project from here use /alemax:new-project."
        )
        return out
    if not (root / ".meta-version").is_file():
        out["blocked"].append(
            "no .meta-version — this is not a claude-meta-bootstrapped project. "
            "Retrofitting an existing repo is a meta-side operation (retrofit-project.sh)."
        )
        return out
    if commits > fresh_max and not force:
        out["blocked"].append(
            f"{commits} commits — this looks like an established project, not a fresh bootstrap. "
            "complete-init's writes (.python-version, settings.local.json, the CLAUDE.md nudge, "
            "ci.yml) can clobber choices the project now owns. Re-run with --force if you mean it."
        )
    return out


def foreign_ci_jobs(root: Path, stack: str) -> list[str]:
    ci = root / ".github" / "workflows" / "ci.yml"
    if not ci.is_file():
        return []
    lines = ci.read_text(encoding="utf-8").splitlines()
    bounds: list[tuple[str, int, int]] = []
    for i, line in enumerate(lines):
        if m := JOB_RE.match(line):
            if bounds:
                bounds[-1] = (bounds[-1][0], bounds[-1][1], i)
            bounds.append((m["job"], i, len(lines)))
    out = []
    for job, start, end in bounds:
        body = "\n".join(lines[start:end])
        owners = {s for s, rx in STACK_HINTS.items() if rx.search(body)}
        if owners and stack not in owners:
            out.append(job)
    return out


def gaps(root: Path, stack: str, commits: int) -> list[dict]:
    found: list[dict] = []

    pyproject = root / "pyproject.toml"
    if pyproject.is_file() and not (root / ".python-version").is_file():
        m = REQUIRES_RE.search(pyproject.read_text(encoding="utf-8"))
        want = m[1] if m else "3.13"
        found.append(
            {
                "id": "python-version",
                "open": True,
                "detail": f"no .python-version; pyproject declares {want}. Without it uv resolves latest "
                f"locally AND in CI while the type checker targets {want} — silent drift.",
                "fix": "init.py fix --gap python-version",
            }
        )

    gitignore = root / ".gitignore"
    ignored = gitignore.is_file() and any(
        SETTINGS_IGNORE_RE.search(ln) for ln in gitignore.read_text(encoding="utf-8").splitlines()
    )
    if not ignored:
        found.append(
            {
                "id": "gitignore-settings",
                "open": True,
                "detail": ".claude/settings.local.json is not gitignored — the grant store must never be "
                "committed, and it has to be ignored BEFORE the floor is reconciled into it.",
                "fix": "init.py fix --gap gitignore-settings",
            }
        )

    if (root / ".claude" / "settings-template.json").is_file():
        found.append(
            {
                "id": "reconcile-settings",
                "open": True,
                "detail": "the tracked floor .claude/settings-template.json is present; fold it into the "
                "gitignored local file (a fresh clone is missing every entry).",
                "fix": "uv run --script bin/reconcile-settings.py check, then apply",
            }
        )

    if foreign := foreign_ci_jobs(root, stack):
        found.append(
            {
                "id": "ci-trim",
                "open": True,
                "detail": f"ci.yml carries jobs for another stack: {', '.join(foreign)}. Runtime gating stops "
                "them running, not Dependabot, which parses the workflow statically and opens bumps "
                "for a toolchain this repo never uses.",
                "fix": f"delete the {', '.join(foreign)} job(s) from .github/workflows/ci.yml",
            }
        )

    if not (root / ".git" / "hooks" / "pre-commit").is_file():
        found.append(
            {
                "id": "pre-commit",
                "open": True,
                "detail": "no .git/hooks/pre-commit — hooks are per-clone and are not cloned.",
                "fix": "pre-commit install",
            }
        )

    if not gitmod.probe(root, "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"):
        found.append(
            {
                "id": "origin-head",
                "open": True,
                "detail": "origin/HEAD is unset; /security-review and other diff-vs-default tools need it.",
                "fix": "git remote set-head origin --auto",
            }
        )

    claude_md = root / "CLAUDE.md"
    if claude_md.is_file() and NUDGE_RE.search(claude_md.read_text(encoding="utf-8")):
        landed = commits > 1
        found.append(
            {
                "id": "bootstrap-nudge",
                "open": True,
                "detail": "the CLAUDE.md bootstrap nudge is still present — remove it only AFTER a first real "
                "change is merged"
                + (
                    "; the history suggests one has landed."
                    if landed
                    else "; nothing but the bootstrap has landed yet, so leave it."
                ),
                "fix": "remove the nudge block by hand once the first change is merged",
            }
        )
    return found


def cmd_context(a) -> int:
    c = context(project_root(), a.force)
    if a.json:
        print(json.dumps(c, indent=2))
    else:
        for b in c["blocked"]:
            print(f"BLOCKED: {b}")
        if not c["blocked"]:
            print(
                f"project={c['project']} stack={c['stack']} commits={c['commits']} "
                f"service={c['service']}" + ("  [forced]" if c["forced"] else "")
            )
    return 1 if c["blocked"] else 0


def cmd_gaps(a) -> int:
    root = project_root()
    c = context(root, force=True)  # gaps is read-only; the gate belongs to `context`
    found = gaps(root, c["stack"], c["commits"])
    if a.json:
        print(json.dumps({"stack": c["stack"], "commits": c["commits"], "gaps": found}, indent=2))
    else:
        for g in found:
            print(f"GAP {g['id']}: {g['detail']}")
            print(f"     fix: {g['fix']}")
        if not found:
            print("no gaps — the bootstrap is complete")
    return 3 if found else 0


def cmd_fix(a) -> int:
    root = project_root()
    if a.gap == "python-version":
        pyproject = root / "pyproject.toml"
        if not pyproject.is_file():
            die("no pyproject.toml — .python-version would pin nothing")
        target = root / ".python-version"
        if target.is_file():
            print(f".python-version already present ({target.read_text().strip()})")
            return 0
        m = REQUIRES_RE.search(pyproject.read_text(encoding="utf-8"))
        want = m[1] if m else "3.13"
        target.write_text(want + "\n", encoding="utf-8")
        print(f"wrote .python-version = {want}")
        return 0
    if a.gap == "gitignore-settings":
        gitignore = root / ".gitignore"
        text = gitignore.read_text(encoding="utf-8") if gitignore.is_file() else ""
        if any(SETTINGS_IGNORE_RE.search(ln) for ln in text.splitlines()):
            print(".claude/settings.local.json is already gitignored")
            return 0
        sep = "" if text.endswith("\n") or not text else "\n"
        gitignore.write_text(
            text + sep + "\n# operator-local Claude Code grants — never committed\n"
            ".claude/settings.local.json\n",
            encoding="utf-8",
        )
        print("added .claude/settings.local.json to .gitignore")
        return 0
    die(f"'{a.gap}' is advisory, not a safe automatic write — run its `fix` line yourself")
    return 1


def register(ap: argparse.ArgumentParser) -> None:
    sub = ap.add_subparsers(dest="subcommand", required=True)
    c = sub.add_parser("context", help="project, .meta-version, and the freshness gate")
    c.add_argument("--force", action="store_true", help="bypass the freshness gate (nothing else)")
    sub.add_parser("gaps", help="what the bootstrap left unfinished")
    f = sub.add_parser("fix", help="apply one safe gap")
    f.add_argument("--gap", required=True, choices=("python-version", "gitignore-settings"))
    # --json belongs on each subcommand, so `<cmd> --json` works (the natural form).
    for sp in sub.choices.values():
        sp.add_argument("--json", action="store_true", help="machine-readable output")

    # Dispatch lives on the subparser, not in a trailing lookup: an unknown
    # subcommand is then argparse's error, with the valid choices, not a KeyError.
    _dispatch = {
        "context": cmd_context,
        "gaps": cmd_gaps,
        "fix": cmd_fix,
    }
    for _name, _sp in sub.choices.items():
        _sp.set_defaults(func=_dispatch[_name])
