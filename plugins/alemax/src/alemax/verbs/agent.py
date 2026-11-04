"""`alemax agent` — start bounded Claude sessions in the right folder (spec `agent-launcher`).

    deliver [--run ID] [--parallel N] [--max-turns N] [--timeout MIN] [--dry-run]
            [--no-post] [--no-tabs]
        For each project a broadcast run delivered to, start a headless session in that
        project's own folder running `/alemax:complete-update` with no human (design D11).
        Wait for all of them, keep each result in this clone's .local/, print a summary,
        post it, and open a Ghostty tab only for a session that needs the operator (D13).

Runs in a claude-meta clone, the one whose `/alemax:update-skills` wrote the run record
(`.local/broadcasts/<run-id>.json`, D10). It never reads a project's repo: each result is
the output of a process this verb started. Each session is the project's own, in its own
folder with its own settings and plugins (Guardrail 4, as amended by D7), and it merges a
clean delivery to the project's `main` itself (D12).

Exit codes: 0 every run ended `done` or `nothing-to-apply` (or a dry run) · 3 a run needs
the operator, timed out or failed, or a delivered project could not be started · 1 not a
claude-meta clone, or no run record.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from ..io import ERROR, FOUND, OK, die
from ..repo import is_meta_repo, toplevel

PROMPT = "/alemax:complete-update"
AUTONOMOUS_ENV = "ALEMAX_AUTONOMOUS"
# The tools a delivery needs: read and edit the checkout, git and gh, re-lock and run the
# project's checks, and the alemax verbs. Anything else is denied in -p, which ends the run
# as needs-operator naming the call (design D11, risks).
ALLOWED_TOOLS = (
    "Read",
    "Edit",
    "Write",
    "Glob",
    "Grep",
    "Bash(git *)",
    "Bash(gh *)",
    "Bash(uv *)",
    "Bash(go *)",
    "Bash(cargo *)",
    "Bash(mvn *)",
    "Bash(make *)",
    "Bash(pre-commit *)",
)
DONE_STATUSES = ("done", "nothing-to-apply")
LINE_STATUSES = ("done", "needs-operator", "nothing-to-apply")


# --- the run record -----------------------------------------------------------------


def runs_dir(root: Path) -> Path:
    return Path(os.environ.get("BROADCAST_RUNS_DIR") or root / ".local" / "broadcasts")


def load_record(root: Path, run_id: str | None) -> dict[str, Any]:
    base = runs_dir(root)
    if run_id:
        path = base / f"{run_id}.json"
    else:
        records = sorted(base.glob("*.json"))
        if not records:
            die(f"no run record in {base} — run /alemax:update-skills first", code=ERROR)
        path = records[-1]
    if not path.is_file():
        die(f"no run record {path}", code=ERROR)
    record = json.loads(path.read_text())
    record["_path"] = str(path)
    return record


def volume_of(path: str) -> str | None:
    parts = Path(os.path.realpath(path)).parts
    return parts[2] if len(parts) > 2 and parts[1] == "Volumes" else None


# --- one session --------------------------------------------------------------------


def claude_command(claude: str, max_turns: int) -> list[str]:
    return [
        claude,
        "-p",
        PROMPT,
        "--output-format",
        "json",
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        " ".join(ALLOWED_TOOLS),
        "--max-turns",
        str(max_turns),
    ]


def final_line(text: str) -> tuple[str, str]:
    """(status, line) from the session's last reply: the last `done|needs-operator|...:` line."""
    for raw in reversed((text or "").strip().splitlines()):
        line = raw.strip().strip("`")
        for status in LINE_STATUSES:
            if line.startswith(status + ":"):
                return status, line
    return "unclear", ""


def run_one(project: dict[str, Any], cmd: list[str], timeout_s: int) -> dict[str, Any]:
    env = {**os.environ, AUTONOMOUS_ENV: "1"}
    vol = volume_of(project["path"])
    if vol:
        env["AIML_VOLUME"] = f"/Volumes/{vol}"
    out: dict[str, Any] = {"project": project["name"], "path": project["path"]}
    try:
        done = subprocess.run(
            cmd, cwd=project["path"], env=env, capture_output=True, text=True, timeout=timeout_s
        )
    except subprocess.TimeoutExpired:
        return {**out, "status": "timeout", "stop_reason": f"past {timeout_s} s"}
    except OSError as exc:
        return {**out, "status": "failed", "stop_reason": f"{type(exc).__name__}: {exc}"}
    out["exit_code"] = done.returncode
    try:
        result = json.loads(done.stdout or "{}")
    except ValueError:
        result = {}
    status, line = final_line(str(result.get("result", "")))
    if result.get("subtype") == "error_max_turns":
        status = "max-turns"
    elif not result:
        status = "failed"
    out.update(
        status=status,
        line=line,
        session_id=result.get("session_id"),
        turns=result.get("num_turns"),
        cost_usd=result.get("total_cost_usd"),
        stop_reason=result.get("subtype"),
        stderr_tail=(done.stderr or "")[-500:],
    )
    return out


# --- the operator's tabs (design D13) -------------------------------------------------


def as_applescript(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def waiting_for_operator(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in results if r.get("status") == "needs-operator" and r.get("session_id")]


def tabs_script(waiting: list[dict[str, Any]], claude: str) -> str:
    """One Ghostty window, a tab per waiting session, each resuming it in its own folder."""
    lines = ['tell application "Ghostty"', "  activate"]
    for i, r in enumerate(waiting):
        cmd = f"{shlex.quote(claude)} --resume {shlex.quote(r['session_id'])}"
        lines += [
            "  set cfg to new surface configuration",
            f"  set initial working directory of cfg to {as_applescript(r['path'])}",
            f"  set command of cfg to {as_applescript(cmd)}",
            "  set w to new window with configuration cfg"
            if i == 0
            else "  new tab in w with configuration cfg",
        ]
    lines.append("end tell")
    return "\n".join(lines) + "\n"


def resume_tabs(results: list[dict[str, Any]], claude: str) -> list[str]:
    """Open the tabs (design D13); with nothing waiting, open nothing."""
    waiting = waiting_for_operator(results)
    if not waiting:
        return []
    script = tabs_script(waiting, claude)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "tabs.applescript"
        src.write_text(script)
        compiled = subprocess.run(
            ["osacompile", "-o", str(Path(tmp) / "tabs.scpt"), str(src)], capture_output=True
        )
        opened = compiled.returncode == 0 and (
            subprocess.run(
                ["osascript", str(Path(tmp) / "tabs.scpt")], capture_output=True
            ).returncode
            == 0
        )
    if not opened:  # an older Ghostty: one window per session, the launchers' proven call
        for r in waiting:
            cmd = f"{shlex.quote(claude)} --resume {shlex.quote(r['session_id'])}"
            subprocess.run(
                [
                    "open",
                    "-na",
                    "Ghostty.app",
                    "--args",
                    f"--working-directory={r['path']}",
                    f"--initial-command={cmd}",
                ],
                capture_output=True,
            )
    return [r["project"] for r in waiting]


# --- deliver ------------------------------------------------------------------------


def summary(record: dict[str, Any], results: list[dict[str, Any]], skipped: list[str]) -> str:
    rows = [f"delivery run {record['run']} — {len(results)} session(s)"]
    for r in results:
        rows.append(
            f"  {r['status']:<17} {r['project']:<40} {r.get('line') or r.get('stop_reason') or ''}"
        )
    for s in skipped:
        rows.append(f"  {'not started':<17} {s}")
    return "\n".join(rows)


def post_summary(root: Path, text: str) -> None:
    """Post from this clone's own account; Mattermost is optional, so a failure is a note."""
    done = subprocess.run(
        [
            "uv",
            "run",
            "--quiet",
            "--project",
            str(root / "plugins" / "alemax"),
            "alemax",
            "presence",
            "post",
            text,
        ],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if done.returncode != 0:
        print(f"note: summary not posted ({(done.stderr or done.stdout).strip()[:200]})")


def cmd_deliver(a: argparse.Namespace) -> int:
    root = toplevel()
    if not is_meta_repo(root):
        die(
            "deliver runs in the claude-meta clone whose broadcast wrote the run record", code=ERROR
        )
    record = load_record(root, a.run)
    claude = a.claude or os.environ.get("ALEMAX_CLAUDE") or "claude"
    here = volume_of(str(root))
    targets, skipped = [], []
    for p in record.get("projects", []):
        if p.get("status") != "delivered":
            continue
        if here and volume_of(p["path"]) not in (None, here):
            skipped.append(f"{p['name']} — on another volume; start it from that volume's clone")
            continue
        if not Path(p["path"]).is_dir():
            skipped.append(f"{p['name']} — {p['path']} is not here")
            continue
        targets.append(p)
    cmd = claude_command(claude, a.max_turns)
    print(
        f"run {record['run']}: {len(targets)} delivered project(s) to complete ({record['_path']})"
    )
    if a.dry_run:
        for p in targets:
            print(f"  would run in {p['path']}: {AUTONOMOUS_ENV}=1 {shlex.join(cmd)}")
        for s in skipped:
            print(f"  would not start: {s}")
        return OK
    if not targets:
        print("nothing to start")
        return OK if not skipped else FOUND

    with ThreadPoolExecutor(max_workers=max(1, a.parallel)) as pool:
        results = list(pool.map(lambda p: run_one(p, cmd, max(1, int(a.timeout * 60))), targets))

    out_dir = runs_dir(root) / record["run"]
    out_dir.mkdir(parents=True, exist_ok=True)
    for r in results:
        (out_dir / f"{r['project']}.json").write_text(json.dumps(r, indent=2) + "\n")

    text = summary(record, results, skipped)
    print(text)
    print(f"results: {out_dir}")
    if not a.no_post:
        post_summary(root, text)
    if not a.no_tabs:
        opened = resume_tabs(results, claude)
        if opened:
            print(
                f"opened a Ghostty tab for {len(opened)} session(s) that need you: {', '.join(opened)}"
            )
    ok = all(r["status"] in DONE_STATUSES for r in results) and not skipped
    return OK if ok else FOUND


# --- registration -------------------------------------------------------------------


def register(ap: argparse.ArgumentParser) -> None:
    subs = ap.add_subparsers(dest="subcommand", required=True, metavar="SUBCOMMAND")
    d = subs.add_parser("deliver", help="complete a broadcast run's deliveries, headless")
    d.add_argument("--run", help="the run id (default: the newest record)")
    d.add_argument("--parallel", type=int, default=4, help="sessions at a time (default 4)")
    d.add_argument("--max-turns", type=int, default=60, help="per session (default 60)")
    d.add_argument("--timeout", type=float, default=30, help="minutes per session (default 30)")
    d.add_argument("--claude", help="the claude binary (default: $ALEMAX_CLAUDE or claude)")
    d.add_argument("--dry-run", action="store_true", help="print the commands; start nothing")
    d.add_argument("--no-post", action="store_true", help="do not post the summary")
    d.add_argument("--no-tabs", action="store_true", help="do not open tabs for questions")
    d.set_defaults(func=cmd_deliver)
