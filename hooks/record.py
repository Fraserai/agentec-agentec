#!/usr/bin/env python3
"""Agentec recorder — the hook handler that captures both sides of the record.

Two independent accounts of every tool call:

  REPORTED  what the agent said it did, from the hook payload
  OBSERVED  what actually changed on disk, from a filesystem scan we run
            ourselves between PreToolUse and PostToolUse

A DIVERGENCE is a file that changed while a tool ran, that the tool never
declared. Write and Edit declare a file_path, so their changes are accounted
for. Bash declares nothing — a command that rewrites twenty files looks
identical in the tool log to one that touches none. That gap is the reason
this exists.

Reads a hook JSON payload on stdin. Never blocks a tool call, never writes to
stdout, and swallows its own errors: a broken flight recorder must not be able
to take down the thing it is recording.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HOME = Path.home()
RUNS = Path(os.environ.get("AGENTEC_HOME", HOME / "Desktop" / "agentec")) / "runs"

# Directories never worth scanning: enormous, noisy, and not where an
# unreported write would actually matter.
SKIP_DIRS = {
    ".git", "node_modules", "__pycache__", ".venv", "venv", ".next",
    "dist", "build", ".cache", "site-packages", ".mypy_cache",
    ".pytest_cache", "runs", ".DS_Store",
}

MAX_SCAN_SECONDS = 3.0
MAX_HASH_BYTES = 2_000_000


def log_path(session_id: str) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    return RUNS / f"{session_id or 'unknown'}.jsonl"


def state_path(session_id: str, tool_use_id: str) -> Path:
    RUNS.mkdir(parents=True, exist_ok=True)
    return RUNS / f".pending-{session_id}-{tool_use_id}.json"


def emit(session_id: str, record: dict) -> None:
    with log_path(session_id).open("a") as f:
        f.write(json.dumps(record) + "\n")


def scan(root: Path) -> dict[str, float]:
    """Modification times for every file under root, as a flat map.

    Uses os.scandir rather than shelling out to find: no quoting problems with
    unusual filenames, and it lets us bail out on a deadline. A recorder that
    stalls the agent for ten seconds per tool call will simply be turned off,
    so the deadline matters more than completeness.
    """
    seen: dict[str, float] = {}
    deadline = time.monotonic() + MAX_SCAN_SECONDS
    stack = [root]
    while stack:
        if time.monotonic() > deadline:
            break
        current = stack.pop()
        try:
            with os.scandir(current) as entries:
                for entry in entries:
                    if entry.name in SKIP_DIRS or entry.name.startswith(".pending-"):
                        continue
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(Path(entry.path))
                        elif entry.is_file(follow_symlinks=False):
                            seen[entry.path] = entry.stat().st_mtime
                    except OSError:
                        continue
        except (OSError, PermissionError):
            continue
    return seen


def digest(path: str) -> str | None:
    try:
        if os.path.getsize(path) > MAX_HASH_BYTES:
            return "too-large"
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        return None


def declared_paths(tool_name: str, tool_input: dict) -> list[str]:
    """Files this tool told us it was going to touch.

    Only Write, Edit and NotebookEdit make a claim. Everything else declares
    nothing, so every change during those calls is by definition unreported.
    """
    if tool_name not in {"Write", "Edit", "NotebookEdit", "MultiEdit"}:
        return []
    out = []
    for key in ("file_path", "notebook_path"):
        value = tool_input.get(key)
        if isinstance(value, str):
            out.append(os.path.realpath(value))
    for edit in tool_input.get("edits", []) or []:
        value = (edit or {}).get("file_path")
        if isinstance(value, str):
            out.append(os.path.realpath(value))
    return out


def handle_pre(payload: dict) -> None:
    session_id = payload.get("session_id", "")
    tool_use_id = payload.get("tool_use_id", "")
    cwd = payload.get("cwd") or os.getcwd()
    state = {
        "started": time.time(),
        "cwd": cwd,
        "tool_name": payload.get("tool_name"),
        "tool_input": payload.get("tool_input", {}),
        "before": scan(Path(cwd)),
    }
    state_path(session_id, tool_use_id).write_text(json.dumps(state))


def handle_post(payload: dict, failed: bool = False) -> None:
    session_id = payload.get("session_id", "")
    tool_use_id = payload.get("tool_use_id", "")
    pending = state_path(session_id, tool_use_id)

    try:
        state = json.loads(pending.read_text())
    except (OSError, json.JSONDecodeError):
        state = {}
    finally:
        pending.unlink(missing_ok=True)

    cwd = state.get("cwd") or payload.get("cwd") or os.getcwd()
    before = state.get("before", {})
    after = scan(Path(cwd))

    created = sorted(set(after) - set(before))
    modified = sorted(p for p in set(after) & set(before) if after[p] != before[p])
    deleted = sorted(set(before) - set(after))

    tool_name = payload.get("tool_name") or state.get("tool_name")
    tool_input = payload.get("tool_input") or state.get("tool_input", {})
    declared = {os.path.realpath(p) for p in declared_paths(tool_name, tool_input)}

    touched = [(p, "created") for p in created]
    touched += [(p, "modified") for p in modified]
    touched += [(p, "deleted") for p in deleted]

    divergences = [
        {"path": p, "change": kind, "hash": digest(p) if kind != "deleted" else None}
        for p, kind in touched
        if os.path.realpath(p) not in declared
    ]

    started = state.get("started", time.time())
    emit(session_id, {
        "event": "tool",
        "tool_use_id": tool_use_id,
        "tool_name": tool_name,
        "ts": started,
        "latency_ms": round((time.time() - started) * 1000),
        "status": "failed" if failed else "success",
        "cwd": cwd,
        "reported": {
            "summary": summarise(tool_name, tool_input),
            "declared_paths": sorted(declared),
        },
        "observed": {
            "created": len(created),
            "modified": len(modified),
            "deleted": len(deleted),
        },
        "divergences": divergences,
    })


def summarise(tool_name: str | None, tool_input: dict) -> str:
    """One line describing what the agent said it was doing."""
    if tool_name == "Bash":
        return (tool_input.get("description") or tool_input.get("command", ""))[:200]
    for key in ("file_path", "notebook_path", "pattern", "url", "query", "prompt"):
        value = tool_input.get(key)
        if isinstance(value, str):
            return value[:200]
    return ""


def handle_session(payload: dict, kind: str) -> None:
    emit(payload.get("session_id", ""), {
        "event": kind,
        "ts": time.time(),
        "cwd": payload.get("cwd"),
        "detail": payload.get("how") or payload.get("why") or "",
    })


def handle_prompt(payload: dict) -> None:
    emit(payload.get("session_id", ""), {
        "event": "user_prompt",
        "ts": time.time(),
        "text": (payload.get("user_input") or "")[:500],
    })


def main() -> None:
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return

    event = payload.get("hook_event_name", "")
    if event == "PreToolUse":
        handle_pre(payload)
    elif event == "PostToolUse":
        handle_post(payload)
    elif event == "PostToolUseFailure":
        handle_post(payload, failed=True)
    elif event in ("SessionStart", "SessionEnd"):
        handle_session(payload, event.replace("Session", "session_").lower())
    elif event == "UserPromptSubmit":
        handle_prompt(payload)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # A flight recorder that can crash the flight is worse than no recorder.
        pass
    sys.exit(0)
