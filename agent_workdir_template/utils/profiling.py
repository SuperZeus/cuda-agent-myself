#!/usr/bin/env python3
"""Phase-0 profiling for eager, optional torch.compile, and candidate model."""

from __future__ import annotations

import statistics
import time
from pathlib import Path

try:
    import torch
except Exception:
    torch = None


def to_sequence(value):
    return value if isinstance(value, (list, tuple)) else [value]


def transform_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (list, tuple)):
        return type(value)(transform_tensors(x, fn) for x in value)
    if isinstance(value, dict):
        return {k: transform_tensors(v, fn) for k, v in value.items()}
    return value


def benchmark(model, inputs, device: str, iters: int = 20, warmup: int = 5) -> dict:
    samples = []
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(*inputs)
        if device == "cuda":
            torch.cuda.synchronize()
        for _ in range(iters):
            start = time.perf_counter()
            _ = model(*inputs)
            if device == "cuda":
                torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1e6)
    return {
        "median_us": round(statistics.median(samples), 3),
        "mean_us": round(statistics.mean(samples), 3),
        "stdev_us": round(statistics.pstdev(samples), 3),
    }


def main() -> int:
    if torch is None:
        print("[SKIP] torch is not installed; profiling downgraded to structural preflight")
        required = ["model.py", "model_new.py", "inputs.py", "task.json"]
        missing = [name for name in required if not Path(name).exists()]
        if missing:
            print(f"[FAIL] missing required files: {', '.join(missing)}")
            return 1
        print("[PASS] profile preflight success")
        return 0

    from inputs import get_init_inputs, get_inputs
    from model import Model
    from model_new import ModelNew

    device = "cuda" if torch.cuda.is_available() else "cpu"
    init_inputs = transform_tensors(to_sequence(get_init_inputs()), lambda x: x.to(device))
    eager_model = Model(*init_inputs).eval().to(device)
    candidate_model = ModelNew(*init_inputs).eval().to(device)
    candidate_model.load_state_dict(eager_model.state_dict())
    sample_inputs = transform_tensors(to_sequence(get_inputs(seed=7)), lambda x: x.to(device))

    eager_stats = benchmark(eager_model, sample_inputs, device)
    candidate_stats = benchmark(candidate_model, sample_inputs, device)
    print(f"Eager: {eager_stats}")

    try:
        compiled_model = torch.compile(eager_model)
        compile_stats = benchmark(compiled_model, sample_inputs, device)
        print(f"TorchCompile: {compile_stats}")
    except Exception as exc:
        compile_stats = {"error": str(exc)}
        print(f"TorchCompile: unavailable ({exc})")

    print(f"Candidate: {candidate_stats}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
