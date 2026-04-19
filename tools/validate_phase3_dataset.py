#!/usr/bin/env python3
"""Validate generated phase-3 dataset artifacts."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
ALLOWED_POLICIES = {
    "agent_edit_loop_v1",
    "baseline_reference",
    "reference_full_loop",
    "handcrafted_bootstrap",
    "handcrafted_full_loop",
}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    processed = REPO_ROOT / "dataset" / "processed"
    splits = REPO_ROOT / "dataset" / "splits"

    generated = read_jsonl(processed / "generated_tasks.jsonl")
    filtered = read_jsonl(processed / "filtered_tasks.jsonl")
    split_manifest = json.loads((splits / "phase3_split_manifest.json").read_text())
    anti_leakage = json.loads((processed / "anti_leakage_report.json").read_text())
    trajectories = REPO_ROOT / "dataset" / "trajectories"
    success_trajectories_path = REPO_ROOT / "dataset" / "trajectories" / "phase3_success_trajectories.jsonl"
    success_trajectories = read_jsonl(success_trajectories_path) if success_trajectories_path.exists() else []
    rollout_summary_path = trajectories / "phase3_rollout_summary.json"
    rollout_manifest_path = trajectories / "phase3_rollout_manifest.json"
    rollout_summary = json.loads(rollout_summary_path.read_text()) if rollout_summary_path.exists() else {}
    rollout_manifest = json.loads(rollout_manifest_path.read_text()) if rollout_manifest_path.exists() else []

    errors: list[str] = []
    if len(filtered) < 380:
        errors.append(f"filtered dataset too small: {len(filtered)}")
    if len(split_manifest["splits"]["test"]) != 30:
        errors.append("test split must contain 30 tasks")
    if len(split_manifest["splits"]["hidden_test"]) != 20:
        errors.append("hidden_test split must contain 20 tasks")
    if anti_leakage["rules"]["ast_similarity_threshold"] != 0.9:
        errors.append("unexpected AST similarity threshold")
    if success_trajectories and len(success_trajectories) < 100:
        errors.append(f"successful trajectories below target: {len(success_trajectories)}")
    if rollout_summary and rollout_summary.get("trajectory_format") != "phase2_episode_v1":
        errors.append("phase3 trajectories are not using phase2 episode format")
    observed_policies = {episode.get("policy_name") for episode in rollout_manifest if isinstance(episode, dict)}
    if observed_policies and not observed_policies.issubset(ALLOWED_POLICIES):
        errors.append(f"unexpected policies in rollout manifest: {sorted(observed_policies - ALLOWED_POLICIES)}")
    if rollout_manifest and any("steps" not in episode or "step_count" not in episode for episode in rollout_manifest):
        errors.append("rollout manifest is missing episode-level step data")
    if rollout_manifest and len({episode.get('task_id') for episode in rollout_manifest}) < 10:
        errors.append("rollout manifest does not cover all starter tasks")

    summary = {
        "generated_count": len(generated),
        "filtered_count": len(filtered),
        "split_sizes": anti_leakage["split_sizes"],
        "success_trajectory_count": len(success_trajectories),
        "trajectory_format": rollout_summary.get("trajectory_format"),
        "observed_policies": sorted(observed_policies),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
