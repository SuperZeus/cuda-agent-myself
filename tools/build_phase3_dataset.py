#!/usr/bin/env python3
"""Build a phase-3 dataset, freeze split manifests, and emit anti-leakage reports."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from task_generators.build_dataset import build_dataset


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def freeze_splits(filtered_rows: list[dict]) -> dict:
    buckets: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in sorted(filtered_rows, key=lambda item: item["dataset_task_id"]):
        key = (row["family"], row["difficulty"], row["source_template"])
        buckets[key].append(row)

    def allocate(target: int) -> list[dict]:
        selected: list[dict] = []
        bucket_keys = sorted(buckets)
        while len(selected) < target:
            progressed = False
            for key in bucket_keys:
                rows = buckets[key]
                if rows and len(selected) < target:
                    selected.append(rows.pop(0))
                    progressed = True
            if not progressed:
                break
        return selected

    hidden_test = allocate(20)
    test = allocate(30)
    dev = allocate(30)
    train = allocate(300)

    split_manifest = {
        "version": "phase3-v1",
        "splits": {
            "train": [row["dataset_task_id"] for row in train],
            "dev": [row["dataset_task_id"] for row in dev],
            "test": [row["dataset_task_id"] for row in test],
            "hidden_test": [row["dataset_task_id"] for row in hidden_test],
        },
    }
    return split_manifest


def leakage_report(filtered_rows: list[dict], split_manifest: dict) -> dict:
    row_map = {row["dataset_task_id"]: row for row in filtered_rows}
    families = {}
    for split_name, ids in split_manifest["splits"].items():
        families[split_name] = {}
        for task_id in ids:
            family = row_map[task_id]["family"]
            families[split_name][family] = families[split_name].get(family, 0) + 1

    return {
        "version": split_manifest["version"],
        "split_sizes": {name: len(ids) for name, ids in split_manifest["splits"].items()},
        "family_distribution": families,
        "difficulty_distribution": {
            split_name: {
                level: sum(1 for task_id in ids if row_map[task_id]["difficulty"] == level)
                for level in ["L1", "L2", "L3"]
            }
            for split_name, ids in split_manifest["splits"].items()
        },
        "rules": {
            "max_composition_size": 5,
            "runtime_window_ms": [1, 100],
            "ast_similarity_threshold": 0.9,
            "fixed_eval_sets": True,
        },
    }


def main() -> int:
    processed_dir = REPO_ROOT / "dataset" / "processed"
    splits_dir = REPO_ROOT / "dataset" / "splits"
    processed_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    report = build_dataset(processed_dir)
    filtered_rows = read_jsonl(processed_dir / "filtered_tasks.jsonl")
    split_manifest = freeze_splits(filtered_rows)
    leakage = leakage_report(filtered_rows, split_manifest)

    (splits_dir / "phase3_split_manifest.json").write_text(json.dumps(split_manifest, indent=2) + "\n")
    (processed_dir / "anti_leakage_report.json").write_text(json.dumps(leakage, indent=2) + "\n")

    print(json.dumps({"filter_report": report, "split_manifest": split_manifest, "anti_leakage_report": leakage}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
