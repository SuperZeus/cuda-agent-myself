from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from observation.build_observation import build_observation
from runner.agent_tools import copy_handcrafted_variant, list_files, read_file, run_workspace_action, write_file
from runner.instantiate_task import instantiate
from runner.task_registry import load_task


@dataclass(frozen=True)
class AgentPolicy:
    name: str
    bootstrap_mode: str
    include_profile: bool
    max_steps: int = 8


POLICIES = {
    "baseline_reference": AgentPolicy("baseline_reference", "none", False),
    "reference_full_loop": AgentPolicy("reference_full_loop", "none", True),
    "handcrafted_bootstrap": AgentPolicy("handcrafted_bootstrap", "handcrafted", False),
    "handcrafted_full_loop": AgentPolicy("handcrafted_full_loop", "handcrafted", True),
    "agent_edit_loop_v1": AgentPolicy("agent_edit_loop_v1", "edit_loop", True, 12),
}


EDIT_TEMPLATES = {
    "axpby_scalar": {
        "safe": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor):
        super().__init__()
        self.alpha = nn.Parameter(alpha.clone())
        self.beta = nn.Parameter(beta.clone())

    def forward(self, x, y):
        return self.alpha * x + self.beta * y
""",
        "tuned": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, alpha: torch.Tensor, beta: torch.Tensor):
        super().__init__()
        self.alpha = nn.Parameter(alpha.clone())
        self.beta = nn.Parameter(beta.clone())

    def forward(self, x, y):
        x = x.contiguous()
        y = y.contiguous()
        return torch.add(x * self.alpha, y * self.beta)
""",
    },
    "broadcast_add": {
        "safe": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self.bias = nn.Parameter(bias.clone())

    def forward(self, x):
        return x + self.bias
""",
        "tuned": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self.bias = nn.Parameter(bias.clone())

    def forward(self, x):
        x = x.contiguous()
        return torch.add(x, self.bias)
""",
    },
    "clamp_shift": {
        "safe": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, offset: torch.Tensor, gain: torch.Tensor):
        super().__init__()
        self.offset = nn.Parameter(offset.clone())
        self.gain = nn.Parameter(gain.clone())

    def forward(self, x):
        return torch.clamp(x + self.offset, -1.0, 1.0) * self.gain
""",
        "tuned": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, offset: torch.Tensor, gain: torch.Tensor):
        super().__init__()
        self.offset = nn.Parameter(offset.clone())
        self.gain = nn.Parameter(gain.clone())

    def forward(self, x):
        shifted = torch.add(x.contiguous(), self.offset)
        clamped = torch.clamp(shifted, -1.0, 1.0)
        return torch.mul(clamped, self.gain)
""",
    },
    "mean_reduction_lastdim": {
        "safe": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.mean(dim=-1)
""",
        "tuned": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous()
        return torch.sum(x, dim=-1) / x.shape[-1]
""",
    },
    "normalize_l2": {
        "safe": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, eps: torch.Tensor):
        super().__init__()
        self.eps = nn.Parameter(eps.clone())

    def forward(self, x):
        denom = torch.sqrt((x * x).sum(dim=-1, keepdim=True) + self.eps)
        return x / denom
""",
        "tuned": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, eps: torch.Tensor):
        super().__init__()
        self.eps = nn.Parameter(eps.clone())

    def forward(self, x):
        x = x.contiguous()
        inv = torch.rsqrt(torch.sum(x * x, dim=-1, keepdim=True) + self.eps)
        return x * inv
""",
    },
    "relu_bias": {
        "safe": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self.bias = nn.Parameter(bias.clone())

    def forward(self, x):
        return torch.relu(x + self.bias)
""",
        "tuned": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, bias: torch.Tensor):
        super().__init__()
        self.bias = nn.Parameter(bias.clone())

    def forward(self, x):
        shifted = torch.add(x.contiguous(), self.bias)
        return torch.relu(shifted)
""",
    },
    "sigmoid_mul": {
        "safe": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
""",
        "tuned": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous()
        return torch.mul(x, torch.sigmoid(x))
""",
    },
    "softmax_rows": {
        "safe": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.softmax(x, dim=-1)
""",
        "tuned": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x.contiguous()
        shifted = x - x.amax(dim=-1, keepdim=True)
        exp_x = torch.exp(shifted)
        return exp_x / exp_x.sum(dim=-1, keepdim=True)
""",
    },
    "sum_reduction_lastdim": {
        "safe": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.sum(dim=-1)
""",
        "tuned": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sum(x.contiguous(), dim=-1)
""",
    },
    "tanh_residual": {
        "safe": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self.scale = nn.Parameter(scale.clone())

    def forward(self, x):
        return x + self.scale * torch.tanh(x)
""",
        "tuned": """import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self, scale: torch.Tensor):
        super().__init__()
        self.scale = nn.Parameter(scale.clone())

    def forward(self, x):
        x = x.contiguous()
        return torch.add(x, self.scale * torch.tanh(x))
""",
    },
}


def score_progress(step_results: list[dict]) -> dict:
    verify_pass = any(step.get("action_result", {}).get("action") == "verify" and step.get("status") == "passed" for step in step_results)
    profile_step = next(
        (step for step in reversed(step_results) if step.get("action_result", {}).get("action") == "profile" and step.get("status") == "passed"),
        None,
    )
    return {
        "verify_pass": verify_pass,
        "profile_metrics": profile_step.get("action_result", {}).get("profile_metrics") if profile_step else None,
    }


def reward_from_results(step_results: list[dict]) -> int:
    if any(step.get("status") == "failed" for step in step_results):
        return -1
    verify_pass = any(step.get("action_result", {}).get("action") == "verify" and step.get("status") == "passed" for step in step_results)
    if not verify_pass:
        return 0
    profile_step = next(
        (step for step in reversed(step_results) if step.get("action_result", {}).get("action") == "profile" and step.get("status") == "passed"),
        None,
    )
    if profile_step is None:
        return 1
    metrics = profile_step["action_result"].get("profile_metrics") or {}
    eager = metrics.get("SpeedupVsEager", 0.0)
    compile_speed = metrics.get("SpeedupVsTorchCompile", 0.0)
    if eager > 1.05 and compile_speed > 1.05:
        return 3
    if eager > 1.05:
        return 2
    return 1


def build_edit_candidate(task_id: str, stage: str) -> str:
    return EDIT_TEMPLATES[task_id][stage]


def write_model_candidate(task: dict, workdir: Path, steps: list[dict], append_step, stage: str, reason: str) -> None:
    candidate = build_edit_candidate(task["task_id"], stage)
    result = write_file(workdir, "model_new.py", candidate, json.loads((workdir / "workspace_meta.json").read_text())["editable_files"])
    append_step(
        f"edit_model_new_{stage}",
        {
            "action": "write_file",
            "status": "passed",
            "exit_code": 0,
            "stdout_tail": f"wrote model_new.py stage={stage} reason={reason}",
            "stderr_tail": "",
            "duration_sec": 0.0,
            "timed_out": False,
            "run_root": str(workdir.parent),
            "profile_metrics": None,
        },
        [result["path"]],
    )


def run_episode(task_id: str, policy_name: str = "baseline_reference") -> dict:
    task = load_task(task_id)
    policy = POLICIES[policy_name]
    workdir = instantiate(task_id, "reference")
    run_root = workdir.parent
    workspace_meta = json.loads((workdir / "workspace_meta.json").read_text())

    steps: list[dict] = []
    best_result = None

    def append_step(action_name: str, result: dict | None = None, files_touched: list[str] | None = None) -> None:
        nonlocal best_result
        status = result["status"] if result else "passed"
        step = {
            "step_id": len(steps) + 1,
            "observation": build_observation(task, workdir, steps, best_result),
            "action": {"name": action_name},
            "action_result": result,
            "files_touched": files_touched or [],
            "status": status,
            "exit_code": None if result is None else result["exit_code"],
            "stdout_tail": "" if result is None else result["stdout_tail"],
            "stderr_tail": "" if result is None else result["stderr_tail"],
        }
        steps.append(step)
        progress = score_progress(steps)
        if progress["verify_pass"]:
            best_result = progress

    append_step("list_files", {"action": "list_files", "status": "passed", "exit_code": 0, "stdout_tail": "\n".join(list_files(workdir)[:20]), "stderr_tail": "", "duration_sec": 0.0, "timed_out": False, "run_root": str(run_root), "profile_metrics": None})
    append_step("read_model", {"action": "read_file", "status": "passed", "exit_code": 0, "stdout_tail": read_file(workdir, "model.py")[:800], "stderr_tail": "", "duration_sec": 0.0, "timed_out": False, "run_root": str(run_root), "profile_metrics": None})
    append_step("read_model_new", {"action": "read_file", "status": "passed", "exit_code": 0, "stdout_tail": read_file(workdir, "model_new.py")[:800], "stderr_tail": "", "duration_sec": 0.0, "timed_out": False, "run_root": str(run_root), "profile_metrics": None})

    if policy.bootstrap_mode == "handcrafted":
        copied = copy_handcrafted_variant(task_id, workdir, workspace_meta["editable_files"])
        append_step("bootstrap_handcrafted", {"action": "write_patch", "status": "passed", "exit_code": 0, "stdout_tail": f"copied {len(copied)} files", "stderr_tail": "", "duration_sec": 0.0, "timed_out": False, "run_root": str(run_root), "profile_metrics": None}, copied)
    elif policy.bootstrap_mode == "edit_loop":
        write_model_candidate(task, workdir, steps, append_step, "safe", "replace placeholder alias with explicit candidate")

    compile_result = run_workspace_action(task, workdir, "compile")
    append_step("run_compile", compile_result)
    if compile_result["status"] == "failed":
        if policy.bootstrap_mode == "edit_loop":
            write_model_candidate(task, workdir, steps, append_step, "safe", "compile failed, revert to safe explicit model")
            compile_result = run_workspace_action(task, workdir, "compile")
            append_step("rerun_compile", compile_result)
            if compile_result["status"] == "failed":
                return finalize_episode(task, policy, run_root, workdir, steps, "compile_failed")
        else:
            return finalize_episode(task, policy, run_root, workdir, steps, "compile_failed")

    verify_result = run_workspace_action(task, workdir, "verify")
    append_step("run_verify", verify_result)
    if verify_result["status"] != "passed":
        if policy.bootstrap_mode == "edit_loop":
            write_model_candidate(task, workdir, steps, append_step, "safe", "verify failed, fallback to safe candidate")
            compile_result = run_workspace_action(task, workdir, "compile")
            append_step("fallback_compile", compile_result)
            verify_result = run_workspace_action(task, workdir, "verify")
            append_step("fallback_verify", verify_result)
            if verify_result["status"] != "passed":
                return finalize_episode(task, policy, run_root, workdir, steps, "verify_failed")
        else:
            return finalize_episode(task, policy, run_root, workdir, steps, "verify_failed")

    if policy.include_profile:
        profile_result = run_workspace_action(task, workdir, "profile")
        append_step("run_profile", profile_result)
        if policy.bootstrap_mode == "edit_loop":
            metrics = profile_result.get("profile_metrics") or {}
            if metrics.get("SpeedupVsEager", 0.0) <= 1.05 and len(steps) < policy.max_steps:
                previous_eager_speedup = metrics.get("SpeedupVsEager", 0.0)
                write_model_candidate(task, workdir, steps, append_step, "tuned", "profile shows weak eager speedup, try tuned rewrite")
                compile_result = run_workspace_action(task, workdir, "compile")
                append_step("tuned_compile", compile_result)
                if compile_result["status"] != "failed":
                    verify_result = run_workspace_action(task, workdir, "verify")
                    append_step("tuned_verify", verify_result)
                    if verify_result["status"] == "passed":
                        profile_result = run_workspace_action(task, workdir, "profile")
                        append_step("tuned_profile", profile_result)
                        tuned_metrics = profile_result.get("profile_metrics") or {}
                        if tuned_metrics.get("SpeedupVsEager", 0.0) < previous_eager_speedup:
                            write_model_candidate(task, workdir, steps, append_step, "safe", "tuned candidate regressed profile, restore safer faster candidate")
                            compile_result = run_workspace_action(task, workdir, "compile")
                            append_step("regression_restore_compile", compile_result)
                            verify_result = run_workspace_action(task, workdir, "verify")
                            append_step("regression_restore_verify", verify_result)
                            if verify_result["status"] == "passed":
                                profile_result = run_workspace_action(task, workdir, "profile")
                                append_step("regression_restore_profile", profile_result)
                    else:
                        write_model_candidate(task, workdir, steps, append_step, "safe", "tuned candidate failed verify, restore safe candidate")
                        compile_result = run_workspace_action(task, workdir, "compile")
                        append_step("restore_compile", compile_result)
                        verify_result = run_workspace_action(task, workdir, "verify")
                        append_step("restore_verify", verify_result)
                        if verify_result["status"] == "passed":
                            profile_result = run_workspace_action(task, workdir, "profile")
                            append_step("restore_profile", profile_result)

    return finalize_episode(task, policy, run_root, workdir, steps, "success")


def finalize_episode(task: dict, policy: AgentPolicy, run_root: Path, workdir: Path, steps: list[dict], termination_reason: str) -> dict:
    reward = reward_from_results(steps)
    episode = {
        "task_id": task["task_id"],
        "policy_name": policy.name,
        "termination_reason": termination_reason,
        "success": reward >= 1,
        "final_reward": reward,
        "run_root": str(run_root),
        "step_count": len(steps),
        "steps": steps,
    }
    (run_root / "agent_trajectory.json").write_text(json.dumps(episode, indent=2) + "\n")
    (run_root / "agent_summary.json").write_text(
        json.dumps(
            {
                "task_id": task["task_id"],
                "policy_name": policy.name,
                "termination_reason": termination_reason,
                "success": episode["success"],
                "final_reward": reward,
                "step_count": len(steps),
            },
            indent=2,
        )
        + "\n"
    )
    return episode
