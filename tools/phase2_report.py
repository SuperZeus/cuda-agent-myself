#!/usr/bin/env python3
"""Run a compact phase-2 report across the starter tasks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.agent_episode import run_episode


POLICY_BY_TASK = {
    "axpby_scalar": "agent_edit_loop_v1",
    "broadcast_add": "agent_edit_loop_v1",
    "relu_bias": "agent_edit_loop_v1",
    "sigmoid_mul": "agent_edit_loop_v1",
    "sum_reduction_lastdim": "agent_edit_loop_v1",
    "mean_reduction_lastdim": "agent_edit_loop_v1",
    "clamp_shift": "agent_edit_loop_v1",
    "normalize_l2": "agent_edit_loop_v1",
    "softmax_rows": "agent_edit_loop_v1",
    "tanh_residual": "agent_edit_loop_v1",
}


def main() -> int:
    episodes = []
    for task_id, policy in POLICY_BY_TASK.items():
        episodes.append(run_episode(task_id, policy))

    summary = {
        "task_count": len(episodes),
        "success_count": sum(1 for episode in episodes if episode["success"]),
        "success_rate": round(sum(1 for episode in episodes if episode["success"]) / len(episodes), 4),
        "avg_reward": round(sum(episode["final_reward"] for episode in episodes) / len(episodes), 4),
        "policy_usage": {
            policy: sum(1 for episode in episodes if episode["policy_name"] == policy)
            for policy in sorted(set(POLICY_BY_TASK.values()))
        },
    }
    print(json.dumps({"summary": summary, "episodes": episodes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
