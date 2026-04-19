#!/usr/bin/env python3
"""Validate standardized tasks for phase-0 repository hygiene."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.task_registry import TASKS_ROOT, list_task_ids, load_task


REQUIRED_FILES = {
    "task.json",
    "metadata.json",
    "reference.md",
    "expected_behavior.md",
    "model.py",
    "inputs.py",
}


def validate_task(task_id: str) -> list[str]:
    errors: list[str] = []
    task_dir = TASKS_ROOT / task_id
    task = load_task(task_id)

    missing = sorted(name for name in REQUIRED_FILES if not (task_dir / name).exists())
    if missing:
        errors.append(f"{task_id}: missing files: {', '.join(missing)}")

    if task["task_id"] != task_id:
        errors.append(f"{task_id}: task_id mismatch in task.json")

    editable = set(task["editable_files"])
    if "model_new.py" not in editable:
        errors.append(f"{task_id}: editable_files must include model_new.py")

    if not any(path.startswith("kernels") for path in editable):
        errors.append(f"{task_id}: editable_files must include kernels/")

    return errors


def main() -> int:
    task_ids = list_task_ids()
    if not task_ids:
        print("[FAIL] no tasks registered")
        return 1

    errors: list[str] = []
    for task_id in task_ids:
        errors.extend(validate_task(task_id))

    summary = {
        "task_count": len(task_ids),
        "tasks": task_ids,
        "errors": errors,
    }
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
