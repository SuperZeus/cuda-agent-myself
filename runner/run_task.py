#!/usr/bin/env python3
"""Instantiate a task and run compile, verify, or profile through sandbox v1."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from runner.instantiate_task import instantiate
from runner.task_registry import REPO_ROOT, load_task


ACTION_TO_SCRIPT = {
    "compile": ["python3", "-m", "utils.compile"],
    "verify": ["python3", "-m", "utils.verification"],
    "profile": ["python3", "-m", "utils.profiling"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--action", choices=sorted(ACTION_TO_SCRIPT), required=True)
    parser.add_argument("--variant", default="reference")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    task = load_task(args.task_id)
    workdir = instantiate(args.task_id, args.variant)
    run_root = workdir.parent
    run_id = run_root.name

    env_snapshot = run_root / "env_snapshot.json"
    subprocess.run(
        ["python3", str(REPO_ROOT / "tools" / "env_snapshot.py"), str(env_snapshot)],
        cwd=REPO_ROOT,
        check=False,
    )

    command = [
        "python3",
        str(REPO_ROOT / "sandbox" / "run_command.py"),
        "--task-id",
        args.task_id,
        "--run-id",
        run_id,
        "--action",
        args.action,
        "--timeout-sec",
        str(task["timeout_limits"][f"{args.action}_sec"]),
        "--workdir",
        str(workdir),
        "--log-dir",
        str(run_root / "logs"),
    ]
    if args.action == "profile":
        command.append("--profile-lock")
    command.append("--")
    command.extend(ACTION_TO_SCRIPT[args.action])

    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    event_path = run_root / "logs" / f"{args.action}.event.json"
    if event_path.exists():
        event = json.loads(event_path.read_text())
        event["env_snapshot_path"] = str(env_snapshot)
        event_path.write_text(json.dumps(event, indent=2) + "\n")
    print(run_root)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
