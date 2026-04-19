#!/usr/bin/env python3
"""Render a compact replay summary for an agent trajectory file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_path")
    args = parser.parse_args()

    path = Path(args.trajectory_path)
    episode = json.loads(path.read_text())
    lines = [
        f"task_id: {episode['task_id']}",
        f"policy: {episode['policy_name']}",
        f"termination: {episode['termination_reason']}",
        f"success: {episode['success']}",
        f"final_reward: {episode['final_reward']}",
        f"step_count: {episode['step_count']}",
        "",
    ]
    for step in episode["steps"]:
        lines.append(
            f"step {step['step_id']}: {step['action']['name']} status={step['status']} exit_code={step['exit_code']}"
        )
        if step.get("files_touched"):
            lines.append(f"  files_touched: {', '.join(step['files_touched'])}")
        if step.get("stdout_tail"):
            lines.append(f"  stdout_tail: {step['stdout_tail'].splitlines()[0]}")
        if step.get("stderr_tail"):
            lines.append(f"  stderr_tail: {step['stderr_tail'].splitlines()[0]}")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
