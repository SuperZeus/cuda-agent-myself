from __future__ import annotations

import ast
import hashlib
import itertools
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from task_generators.catalog import GENERATOR_TEMPLATES, HOLDOUT_SIGNATURES


@dataclass
class DatasetTask:
    dataset_task_id: str
    source_template: str
    family: str
    difficulty: str
    composition_size: int
    shape_bucket: str
    dtype: str
    operation_sequence: list[str]
    metadata: dict
    filters: dict


def ast_signature(sequence: list[str]) -> str:
    expr = "x"
    for op in sequence:
        if op == "add":
            expr = f"({expr}+y)"
        elif op == "mul":
            expr = f"({expr}*y)"
        elif op == "sigmoid":
            expr = f"sigmoid({expr})"
        elif op == "relu":
            expr = f"relu({expr})"
        elif op == "tanh":
            expr = f"tanh({expr})"
        elif op == "clamp":
            expr = f"clamp({expr})"
        elif op == "bias_add":
            expr = f"({expr}+bias)"
        elif op == "sum_lastdim":
            expr = f"sum_lastdim({expr})"
        elif op == "mean_lastdim":
            expr = f"mean_lastdim({expr})"
        elif op == "scale":
            expr = f"({expr}*scale)"
        elif op == "square":
            expr = f"({expr}*{expr})"
        elif op == "sqrt":
            expr = f"sqrt({expr})"
        elif op == "divide":
            expr = f"(x/{expr})"
        else:
            expr = f"{op}({expr})"
    return ast.dump(ast.parse(expr, mode="eval"), include_attributes=False)


def similarity(a: list[str], b: list[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(set(a) & set(b))
    union = len(set(a) | set(b))
    return inter / union if union else 0.0


def estimated_runtime_ms(shape_bucket: str, sequence: list[str]) -> float:
    volume = 1
    for dim in shape_bucket.split("x"):
        volume *= int(dim)
    reduction_bonus = 1.0 + 0.15 * sum(1 for op in sequence if op in {"sum_lastdim", "mean_lastdim", "sqrt", "divide"})
    return round((volume / 3500.0) * (0.22 * len(sequence)) * reduction_bonus, 3)


def generate_tasks() -> list[DatasetTask]:
    results: list[DatasetTask] = []
    for template in GENERATOR_TEMPLATES:
        max_comp = min(template["max_composition"], 5)
        for shape_bucket in template["shapes"]:
            for comp_size in range(2, max_comp + 1):
                for seq in itertools.product(template["ops"], repeat=comp_size):
                    if len(set(seq)) == 1:
                        continue
                    seq_list = list(seq)
                    signature = ast_signature(seq_list)
                    holdout_sim = max(similarity(seq_list, value) for value in HOLDOUT_SIGNATURES.values())
                    runtime_ms = estimated_runtime_ms(shape_bucket, seq_list)
                    task_id = hashlib.sha1(
                        f"{template['template_id']}|{shape_bucket}|{'-'.join(seq_list)}".encode()
                    ).hexdigest()[:12]
                    results.append(
                        DatasetTask(
                            dataset_task_id=f"{template['template_id']}_{task_id}",
                            source_template=template["template_id"],
                            family=template["family"],
                            difficulty=template["difficulty"],
                            composition_size=len(seq_list),
                            shape_bucket=shape_bucket,
                            dtype="float32",
                            operation_sequence=seq_list,
                            metadata={
                                "generator": "phase3-v1",
                                "holdout_bucket": "kernelbench_guard",
                            },
                            filters={
                                "runtime_ms": runtime_ms,
                                "runtime_pass": 1.0 <= runtime_ms <= 100.0,
                                "ast_signature": signature,
                                "ast_similarity_to_holdout": round(holdout_sim, 4),
                                "ast_similarity_pass": holdout_sim < 0.9,
                                "composition_pass": len(seq_list) <= 5,
                            },
                        )
                    )
    return results


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def build_dataset(processed_dir: Path) -> dict:
    generated = generate_tasks()
    generated_rows = [asdict(task) for task in generated]
    write_jsonl(processed_dir / "generated_tasks.jsonl", generated_rows)

    filtered = [
        row
        for row in generated_rows
        if row["filters"]["runtime_pass"]
        and row["filters"]["ast_similarity_pass"]
        and row["filters"]["composition_pass"]
    ]
    write_jsonl(processed_dir / "filtered_tasks.jsonl", filtered)

    report = {
        "generated_count": len(generated_rows),
        "filtered_count": len(filtered),
        "families": sorted({row["family"] for row in filtered}),
        "difficulty_counts": {
            level: sum(1 for row in filtered if row["difficulty"] == level)
            for level in ["L1", "L2", "L3"]
        },
        "template_counts": {
            template["template_id"]: sum(1 for row in filtered if row["source_template"] == template["template_id"])
            for template in GENERATOR_TEMPLATES
        },
    }
    (processed_dir / "filter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
