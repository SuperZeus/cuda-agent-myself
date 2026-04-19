#!/usr/bin/env python3
"""Batch-sample episode-level rollout trajectories using the current runner."""

from __future__ import annotations

import argparse
import ast
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TASK_TO_VARIANT = {
    "axpby_scalar": "handcrafted",
    "broadcast_add": "handcrafted",
    "relu_bias": "handcrafted",
    "sigmoid_mul": "handcrafted",
    "sum_reduction_lastdim": "handcrafted",
    "mean_reduction_lastdim": "handcrafted",
    "clamp_shift": "reference",
    "normalize_l2": "reference",
    "softmax_rows": "reference",
    "tanh_residual": "reference",
}


@dataclass(frozen=True)
class Policy:
    name: str
    variant_mode: str
    actions: tuple[str, ...]
    observation_style: str


POLICIES = [
    Policy("reference_compile_verify", "reference", ("compile", "verify"), "concise"),
    Policy("reference_verify_only", "reference", ("verify",), "minimal"),
    Policy("reference_compile_verify_alt", "reference", ("compile", "verify"), "verbose"),
    Policy("reference_verify_only_alt", "reference", ("verify",), "tail-focused"),
    Policy("reference_compile_then_verify_probe", "reference", ("compile", "verify"), "diagnostic"),
    Policy("preferred_compile_verify", "preferred", ("compile", "verify"), "concise"),
    Policy("preferred_verify_only", "preferred", ("verify",), "minimal"),
    Policy("preferred_compile_verify_alt", "preferred", ("compile", "verify"), "verbose"),
    Policy("preferred_verify_only_alt", "preferred", ("verify",), "tail-focused"),
    Policy("preferred_compile_then_verify_probe", "preferred", ("compile", "verify"), "diagnostic"),
    Policy("preferred_full_loop", "preferred", ("compile", "verify", "profile"), "benchmark"),
]


def choose_variant(task_id: str, policy: Policy) -> str:
    preferred = TASK_TO_VARIANT[task_id]
    if policy.variant_mode == "reference":
        return "reference"
    return preferred


def parse_profile_metrics(stdout_text: str) -> dict:
    metrics: dict = {}
    for raw_line in stdout_text.splitlines():
        line = raw_line.strip()
        if ": " not in line:
            continue
        name, payload = line.split(": ", 1)
        if name in {"Eager", "TorchCompile", "Candidate"}:
            try:
                metrics[name.lower()] = ast.literal_eval(payload)
            except Exception:
                metrics[name.lower()] = {"raw": payload}
        elif name in {"SpeedupVsEager", "SpeedupVsTorchCompile"}:
            try:
                metrics[name] = float(payload)
            except ValueError:
                metrics[name] = payload
    return metrics


def summarize_text(path: Path, lines: int = 10) -> str:
    if not path.exists():
        return ""
    content = path.read_text().splitlines()
    return "\n".join(content[-lines:])


def build_step_from_run_root(run_root: Path, action: str) -> dict:
    event_path = run_root / "logs" / f"{action}.event.json"
    event = json.loads(event_path.read_text())
    stdout_path = Path(event["stdout_path"])
    stderr_path = Path(event["stderr_path"])
    stdout_text = stdout_path.read_text() if stdout_path.exists() else ""
    step = {
        "action": action,
        "exit_code": event["exit_code"],
        "timed_out": event["timed_out"],
        "duration_sec": event["duration_sec"],
        "run_root": str(run_root),
        "stdout_tail": summarize_text(stdout_path),
        "stderr_tail": summarize_text(stderr_path),
        "status": "skipped" if "[SKIP]" in stdout_text else ("passed" if event["exit_code"] == 0 else "failed"),
    }
    if action == "profile":
        step["profile_metrics"] = parse_profile_metrics(stdout_text)
    return step


def find_cached_run(task_id: str, variant: str, action: str) -> Path | None:
    runs_root = REPO_ROOT / "logs" / "runs" / task_id
    if not runs_root.exists():
        return None
    candidates = []
    for run_root in runs_root.iterdir():
        workspace_meta = run_root / "workdir" / "workspace_meta.json"
        event_path = run_root / "logs" / f"{action}.event.json"
        if not workspace_meta.exists() or not event_path.exists():
            continue
        meta = json.loads(workspace_meta.read_text())
        if meta.get("variant") != variant:
            continue
        event = json.loads(event_path.read_text())
        if event["exit_code"] != 0:
            continue
        candidates.append((event["finished_at"], run_root))
    if not candidates:
        return None
    candidates.sort()
    return candidates[-1][1]


def run_action(task_id: str, variant: str, action: str) -> dict:
    cached = find_cached_run(task_id, variant, action)
    if cached is not None:
        step = build_step_from_run_root(cached, action)
        step["cache_hit"] = True
        return step

    cmd = [
        "python3",
        "-m",
        "runner.run_task",
        "--task-id",
        task_id,
        "--variant",
        variant,
        "--action",
        action,
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, check=False)
    run_root = Path(result.stdout.strip().splitlines()[-1]) if result.stdout.strip() else None
    if not run_root:
        return {"action": action, "exit_code": result.returncode, "run_root": None}

    step = build_step_from_run_root(run_root, action)
    step["cache_hit"] = False
    return step


def compute_episode_reward(steps: list[dict]) -> int:
    if any(step["exit_code"] != 0 for step in steps):
        return -1
    verify_steps = [step for step in steps if step["action"] == "verify"]
    if not verify_steps:
        return 0
    profile_steps = [step for step in steps if step["action"] == "profile"]
    if not profile_steps:
        return 1
    metrics = profile_steps[-1].get("profile_metrics", {})
    speedup_eager = metrics.get("SpeedupVsEager", 0.0)
    speedup_compile = metrics.get("SpeedupVsTorchCompile", 0.0)
    if speedup_eager > 1.05 and speedup_compile > 1.05:
        return 3
    if speedup_eager > 1.05:
        return 2
    return 1


def trajectory_success(steps: list[dict]) -> bool:
    has_verify = any(step["action"] == "verify" and step["status"] == "passed" for step in steps)
    all_good = all(step["exit_code"] == 0 for step in steps)
    return has_verify and all_good


def build_episode(task_id: str, policy: Policy, rollout_round: int) -> dict:
    variant = choose_variant(task_id, policy)
    steps = [run_action(task_id, variant, action) for action in policy.actions]
    reward = compute_episode_reward(steps)
    return {
        "episode_id": f"{task_id}:{policy.name}:round{rollout_round}",
        "task_id": task_id,
        "variant": variant,
        "policy_name": policy.name,
        "observation_style": policy.observation_style,
        "rollout_round": rollout_round,
        "steps": steps,
        "final_reward": reward,
        "success": trajectory_success(steps),
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def summarize(episodes: list[dict]) -> dict:
    successful = [episode for episode in episodes if episode["success"]]
    return {
        "episode_count": len(episodes),
        "successful_episode_count": len(successful),
        "successful_task_count": len({episode["task_id"] for episode in successful}),
        "reward_histogram": {
            str(reward): sum(1 for episode in episodes if episode["final_reward"] == reward)
            for reward in [-1, 0, 1, 2, 3]
        },
        "policy_histogram": {
            policy.name: sum(1 for episode in episodes if episode["policy_name"] == policy.name)
            for policy in POLICIES
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=1)
    args = parser.parse_args()

    tasks = list(TASK_TO_VARIANT.keys())[: args.limit]
    episodes: list[dict] = []
    for rollout_round in range(1, args.rounds + 1):
        for task_id in tasks:
            for policy in POLICIES:
                episodes.append(build_episode(task_id, policy, rollout_round))

    trajectories_dir = REPO_ROOT / "dataset" / "trajectories"
    manifest_json = trajectories_dir / "phase3_rollout_manifest.json"
    manifest_jsonl = trajectories_dir / "phase3_rollout_manifest.jsonl"
    success_jsonl = trajectories_dir / "phase3_success_trajectories.jsonl"
    summary_json = trajectories_dir / "phase3_rollout_summary.json"

    manifest_json.write_text(json.dumps(episodes, indent=2) + "\n")
    write_jsonl(manifest_jsonl, episodes)
    write_jsonl(success_jsonl, [episode for episode in episodes if episode["success"]])
    summary_json.write_text(json.dumps(summarize(episodes), indent=2) + "\n")

    print(json.dumps({"manifest": str(manifest_json), "success_manifest": str(success_jsonl), "summary": summarize(episodes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
