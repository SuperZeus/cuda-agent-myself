#!/usr/bin/env python3
"""Instantiate a standardized task into a runnable workdir."""

from __future__ import annotations

import json
import shutil
import stat
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from runner.task_registry import REPO_ROOT, TASKS_ROOT, load_task


PROTECTED_FILES = [
    "utils/verification.py",
    "utils/profiling.py",
    "task.json",
    "workspace_meta.json",
]


def set_readonly(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))


def copy_optional_tree(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    for entry in src.rglob("*"):
        rel = entry.relative_to(src)
        target = dst / rel
        if entry.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(entry, target)


def instantiate(task_id: str, variant: str = "reference") -> Path:
    task = load_task(task_id)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_root = REPO_ROOT / "logs" / "runs" / task_id / run_id
    workdir = run_root / "workdir"
    template = REPO_ROOT / "agent_workdir_template"
    shutil.copytree(template, workdir)

    task_dir = TASKS_ROOT / task_id
    for name in ["model.py", "inputs.py", "task.json", "reference.md", "metadata.json", "expected_behavior.md"]:
        shutil.copy2(task_dir / name, workdir / name)

    if variant != "reference":
        copy_optional_tree(task_dir / variant, workdir)

    workspace_meta = {
        "task_id": task_id,
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "template_version": "phase0-v1",
        "environment": {"profile": "local-dev"},
        "readonly_files": PROTECTED_FILES,
        "editable_files": task["editable_files"],
        "variant": variant,
        "best_reward": None,
        "best_artifact_path": None,
    }
    (workdir / "workspace_meta.json").write_text(json.dumps(workspace_meta, indent=2) + "\n")

    for rel_path in PROTECTED_FILES:
        set_readonly(workdir / rel_path)

    return workdir


def main() -> int:
    if len(sys.argv) not in {2, 3}:
        print("usage: python -m runner.instantiate_task <task-id> [variant]")
        return 1
    variant = sys.argv[2] if len(sys.argv) == 3 else "reference"
    workdir = instantiate(sys.argv[1], variant)
    print(workdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
