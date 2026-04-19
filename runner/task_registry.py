from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TASKS_ROOT = REPO_ROOT / "tasks"


def load_task(task_id: str) -> dict:
    task_dir = TASKS_ROOT / task_id
    task_path = task_dir / "task.json"
    if not task_path.exists():
        raise FileNotFoundError(f"Unknown task: {task_id}")
    return json.loads(task_path.read_text())


def list_task_ids() -> list[str]:
    task_ids = []
    for task_dir in sorted(TASKS_ROOT.iterdir()):
        if task_dir.is_dir() and (task_dir / "task.json").exists():
            task_ids.append(task_dir.name)
    return task_ids
