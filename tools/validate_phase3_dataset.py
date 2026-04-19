#!/usr/bin/env python3
"""Validate generated phase-3 dataset artifacts."""

from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> int:
    processed = REPO_ROOT / "dataset" / "processed"
    splits = REPO_ROOT / "dataset" / "splits"

    generated = read_jsonl(processed / "generated_tasks.jsonl")
    filtered = read_jsonl(processed / "filtered_tasks.jsonl")
    split_manifest = json.loads((splits / "phase3_split_manifest.json").read_text())
    anti_leakage = json.loads((processed / "anti_leakage_report.json").read_text())
    success_trajectories_path = REPO_ROOT / "dataset" / "trajectories" / "phase3_success_trajectories.jsonl"
    success_trajectories = read_jsonl(success_trajectories_path) if success_trajectories_path.exists() else []

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

    summary = {
        "generated_count": len(generated),
        "filtered_count": len(filtered),
        "split_sizes": anti_leakage["split_sizes"],
        "success_trajectory_count": len(success_trajectories),
        "errors": errors,
    }
    print(json.dumps(summary, indent=2))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
