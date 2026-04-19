#!/usr/bin/env python3
"""Sample phase-3 trajectories using the phase-2 episode runner."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.agent_episode import run_episode


TASK_POLICIES = {
    "axpby_scalar": ["baseline_reference", "agent_edit_loop_v1", "handcrafted_bootstrap", "handcrafted_full_loop"],
    "broadcast_add": ["baseline_reference", "agent_edit_loop_v1", "handcrafted_bootstrap", "handcrafted_full_loop"],
    "relu_bias": ["baseline_reference", "agent_edit_loop_v1", "handcrafted_bootstrap", "handcrafted_full_loop"],
    "sigmoid_mul": ["baseline_reference", "agent_edit_loop_v1", "handcrafted_bootstrap", "reference_full_loop"],
    "sum_reduction_lastdim": ["baseline_reference", "agent_edit_loop_v1", "handcrafted_bootstrap", "reference_full_loop"],
    "mean_reduction_lastdim": ["baseline_reference", "agent_edit_loop_v1", "handcrafted_bootstrap", "reference_full_loop"],
    "clamp_shift": ["baseline_reference", "agent_edit_loop_v1", "reference_full_loop"],
    "normalize_l2": ["baseline_reference", "agent_edit_loop_v1", "reference_full_loop"],
    "softmax_rows": ["baseline_reference", "agent_edit_loop_v1", "reference_full_loop"],
    "tanh_residual": ["baseline_reference", "agent_edit_loop_v1", "reference_full_loop"],
}


def summarize(episodes: list[dict]) -> dict:
    success = [episode for episode in episodes if episode["success"]]
    policies = sorted({policy for task_policies in TASK_POLICIES.values() for policy in task_policies})
    return {
        "trajectory_format": "phase2_episode_v1",
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "episode_count": len(episodes),
        "successful_episode_count": len(success),
        "successful_task_count": len({episode["task_id"] for episode in success}),
        "success_rate": round(len(success) / len(episodes), 4) if episodes else 0.0,
        "reward_histogram": {
            str(reward): sum(1 for episode in episodes if episode["final_reward"] == reward)
            for reward in [-1, 0, 1, 2, 3]
        },
        "policy_histogram": {
            policy: sum(1 for episode in episodes if episode["policy_name"] == policy)
            for policy in policies
        },
        "task_histogram": {
            task_id: sum(1 for episode in episodes if episode["task_id"] == task_id)
            for task_id in sorted(TASK_POLICIES)
        },
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args()

    tasks = list(TASK_POLICIES.items())[: args.limit]
    episodes: list[dict] = []
    for rollout_round in range(1, args.rounds + 1):
        for task_id, policies in tasks:
            for policy in policies:
                episode = run_episode(task_id, policy)
                episode["rollout_round"] = rollout_round
                episode["episode_id"] = f"{task_id}:{policy}:round{rollout_round}"
                episodes.append(episode)

    trajectories_dir = REPO_ROOT / "dataset" / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)
    manifest = trajectories_dir / "phase3_rollout_manifest.json"
    manifest_jsonl = trajectories_dir / "phase3_rollout_manifest.jsonl"
    success_manifest = trajectories_dir / "phase3_success_trajectories.jsonl"
    summary_path = trajectories_dir / "phase3_rollout_summary.json"

    manifest.write_text(json.dumps(episodes, indent=2) + "\n")
    write_jsonl(manifest_jsonl, episodes)
    write_jsonl(success_manifest, [episode for episode in episodes if episode["success"]])
    summary = summarize(episodes)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    print(json.dumps({"manifest": str(manifest), "success_manifest": str(success_manifest), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
