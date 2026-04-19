#!/usr/bin/env python3
"""Run phase-1 handcrafted solutions and summarize compile/verify/profile results."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


HANDCRAFTED_TASKS = [
    "axpby_scalar",
    "broadcast_add",
    "relu_bias",
    "sigmoid_mul",
    "sum_reduction_lastdim",
    "mean_reduction_lastdim",
]


def parse_profile_stdout(stdout_path: Path) -> dict:
    data: dict = {}
    if not stdout_path.exists():
        return data

    for raw_line in stdout_path.read_text().splitlines():
        line = raw_line.strip()
        if ": " not in line:
            continue
        name, payload = line.split(": ", 1)
        if name in {"Eager", "TorchCompile", "Candidate"}:
            try:
                data[name.lower()] = ast.literal_eval(payload)
            except Exception:
                data[name.lower()] = {"raw": payload}
        elif name in {"SpeedupVsEager", "SpeedupVsTorchCompile"}:
            try:
                data[name] = float(payload)
            except ValueError:
                data[name] = payload
    return data


def run_action(task_id: str, action: str) -> dict:
    cmd = [
        "python3",
        "-m",
        "runner.run_task",
        "--task-id",
        task_id,
        "--variant",
        "handcrafted",
        "--action",
        action,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    run_root = Path(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else None
    summary = {
        "task_id": task_id,
        "action": action,
        "exit_code": result.returncode,
        "run_root": str(run_root) if run_root else None,
    }
    if not run_root:
        return summary

    event = json.loads((run_root / "logs" / f"{action}.event.json").read_text())
    summary["timed_out"] = event["timed_out"]
    stdout_path = Path(event["stdout_path"])
    stdout_text = stdout_path.read_text() if stdout_path.exists() else ""
    summary["status"] = "skipped" if "[SKIP]" in stdout_text else ("passed" if result.returncode == 0 else "failed")
    if action == "profile":
        summary["profile"] = parse_profile_stdout(stdout_path)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", nargs="+", default=["compile", "verify", "profile"])
    args = parser.parse_args()

    report = []
    for task_id in HANDCRAFTED_TASKS:
        for action in args.actions:
            report.append(run_action(task_id, action))

    compile_rows = [row for row in report if row["action"] == "compile"]
    verify_rows = [row for row in report if row["action"] == "verify"]
    profile_rows = [row for row in report if row["action"] == "profile" and "profile" in row]

    def mean(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 4)

    summary = {
        "task_count": len(HANDCRAFTED_TASKS),
        "compile_pass_count": sum(1 for row in compile_rows if row["status"] == "passed"),
        "compile_skip_count": sum(1 for row in compile_rows if row["status"] == "skipped"),
        "verify_pass_count": sum(1 for row in verify_rows if row["status"] == "passed"),
        "profile_pass_count": sum(1 for row in profile_rows if row["status"] == "passed"),
        "avg_candidate_median_us": mean(
            [
                row["profile"]["candidate"]["median_us"]
                for row in profile_rows
                if "candidate" in row["profile"] and "median_us" in row["profile"]["candidate"]
            ]
        ),
        "avg_speedup_vs_eager": mean(
            [row["profile"]["SpeedupVsEager"] for row in profile_rows if "SpeedupVsEager" in row["profile"]]
        ),
        "avg_speedup_vs_torch_compile": mean(
            [
                row["profile"]["SpeedupVsTorchCompile"]
                for row in profile_rows
                if "SpeedupVsTorchCompile" in row["profile"]
            ]
        ),
    }

    print(json.dumps({"summary": summary, "details": report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
