from __future__ import annotations

import json
from pathlib import Path


def truncate_lines(text: str, limit: int = 12) -> str:
    lines = [line for line in text.splitlines() if line.strip()]
    if len(lines) <= limit:
        return "\n".join(lines)
    return "\n".join(lines[-limit:])


def summarize_action_steps(steps: list[dict], limit: int = 3) -> list[dict]:
    summary = []
    for step in steps[-limit:]:
        summary.append(
            {
                "step_id": step["step_id"],
                "action": step["action"]["name"],
                "status": step.get("status"),
                "exit_code": step.get("exit_code"),
                "files_touched": step.get("files_touched", []),
                "stdout_tail": truncate_lines(step.get("stdout_tail", "")),
                "stderr_tail": truncate_lines(step.get("stderr_tail", "")),
            }
        )
    return summary


def build_observation(task: dict, workdir: Path, steps: list[dict], best_result: dict | None) -> dict:
    workspace_meta = json.loads((workdir / "workspace_meta.json").read_text())
    return {
        "task_id": task["task_id"],
        "goal": task["description"],
        "editable_files": workspace_meta["editable_files"],
        "readonly_files": workspace_meta["readonly_files"],
        "recent_steps": summarize_action_steps(steps),
        "best_result": best_result,
        "current_files": sorted(
            str(path.relative_to(workdir))
            for path in workdir.rglob("*")
            if path.is_file() and ".git" not in path.parts
        )[:80],
    }
