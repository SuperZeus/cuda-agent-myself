from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

from runner.task_registry import REPO_ROOT, TASKS_ROOT


def is_editable(rel_path: str, editable_files: list[str]) -> bool:
    return any(rel_path == allowed or rel_path.startswith(allowed) for allowed in editable_files)


def list_files(workdir: Path) -> list[str]:
    return sorted(
        str(path.relative_to(workdir))
        for path in workdir.rglob("*")
        if path.is_file() and ".git" not in path.parts
    )


def read_file(workdir: Path, rel_path: str) -> str:
    return (workdir / rel_path).read_text()


def write_file(workdir: Path, rel_path: str, content: str, editable_files: list[str]) -> dict:
    if not is_editable(rel_path, editable_files):
        raise PermissionError(f"File is not editable: {rel_path}")
    target = workdir / rel_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"path": rel_path, "bytes": len(content.encode())}


def copy_handcrafted_variant(task_id: str, workdir: Path, editable_files: list[str]) -> list[str]:
    variant_dir = TASKS_ROOT / task_id / "handcrafted"
    if not variant_dir.exists():
        return []

    copied: list[str] = []
    for source in sorted(path for path in variant_dir.rglob("*") if path.is_file()):
        rel_path = str(source.relative_to(variant_dir))
        if not is_editable(rel_path, editable_files):
            continue
        target = workdir / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(rel_path)
    return copied


def run_workspace_action(task: dict, workdir: Path, action: str) -> dict:
    run_root = workdir.parent
    env_snapshot = run_root / "env_snapshot.json"
    subprocess.run(
        ["python3", str(REPO_ROOT / "tools" / "env_snapshot.py"), str(env_snapshot)],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    command = [
        "python3",
        str(REPO_ROOT / "sandbox" / "run_command.py"),
        "--task-id",
        task["task_id"],
        "--run-id",
        run_root.name,
        "--action",
        action,
        "--timeout-sec",
        str(task["timeout_limits"][f"{action}_sec"]),
        "--workdir",
        str(workdir),
        "--log-dir",
        str(run_root / "logs"),
    ]
    if action == "profile":
        command.append("--profile-lock")
    command.append("--")
    command.extend(["python3", "-m", f"utils.{action if action != 'verify' else 'verification'}"])
    if action == "compile":
        command[-1] = "utils.compile"
    elif action == "profile":
        command[-1] = "utils.profiling"

    result = subprocess.run(command, cwd=REPO_ROOT, check=False, capture_output=True, text=True)
    event_path = run_root / "logs" / f"{action}.event.json"
    event = json.loads(event_path.read_text())
    event["env_snapshot_path"] = str(env_snapshot)
    event_path.write_text(json.dumps(event, indent=2) + "\n")

    stdout_path = Path(event["stdout_path"])
    stderr_path = Path(event["stderr_path"])
    stdout_text = stdout_path.read_text() if stdout_path.exists() else ""
    stderr_text = stderr_path.read_text() if stderr_path.exists() else ""

    status = "skipped" if "[SKIP]" in stdout_text else ("passed" if event["exit_code"] == 0 else "failed")
    return {
        "action": action,
        "exit_code": result.returncode,
        "status": status,
        "timed_out": event["timed_out"],
        "duration_sec": event["duration_sec"],
        "stdout_tail": "\n".join(stdout_text.splitlines()[-12:]),
        "stderr_tail": "\n".join(stderr_text.splitlines()[-12:]),
        "run_root": str(run_root),
        "profile_metrics": parse_profile_metrics(stdout_text) if action == "profile" else None,
    }


def parse_profile_metrics(stdout_text: str) -> dict:
    metrics: dict = {}
    for line in stdout_text.splitlines():
        if ": " not in line:
            continue
        name, payload = line.split(": ", 1)
        if name.startswith("Speedup"):
            try:
                metrics[name] = float(payload)
            except ValueError:
                metrics[name] = payload
    return metrics
