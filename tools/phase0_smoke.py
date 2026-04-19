#!/usr/bin/env python3
"""Run phase-0 smoke checks across all registered tasks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.task_registry import REPO_ROOT, list_task_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actions",
        nargs="+",
        default=["compile", "verify"],
        choices=["compile", "verify", "profile"],
        help="Actions to run for each task",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    results = []
    overall_code = 0
    for task_id in list_task_ids():
        for action in args.actions:
            cmd = ["python3", "-m", "runner.run_task", "--task-id", task_id, "--action", action]
            result = subprocess.run(cmd, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
            results.append(
                {
                    "task_id": task_id,
                    "action": action,
                    "exit_code": result.returncode,
                    "run_root": result.stdout.strip().splitlines()[-1] if result.stdout.strip() else None,
                }
            )
            if result.returncode != 0:
                overall_code = 1

    print(json.dumps(results, indent=2))
    return overall_code


if __name__ == "__main__":
    raise SystemExit(main())
