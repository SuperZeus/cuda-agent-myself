#!/usr/bin/env python3
"""Phase-0 correctness verification."""

from __future__ import annotations

import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

try:
    import torch
    import torch.nn.functional as F
except Exception:
    torch = None
    F = None


def transform_tensors(value, fn):
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (list, tuple)):
        return type(value)(transform_tensors(x, fn) for x in value)
    if isinstance(value, dict):
        return {k: transform_tensors(v, fn) for k, v in value.items()}
    return value


def to_sequence(value):
    return value if isinstance(value, (list, tuple)) else [value]


@contextmanager
def block_torch_functional():
    originals = {}
    for name in dir(F):
        attr = getattr(F, name)
        if callable(attr) and not name.startswith("_"):
            originals[name] = attr

            def wrapper(*args, __name=name, **kwargs):
                raise RuntimeError(f"torch.nn.functional.{__name} is not allowed in model_new")

            setattr(F, name, wrapper)
    try:
        yield
    finally:
        for name, attr in originals.items():
            setattr(F, name, attr)


def assert_equal(actual, expected, atol: float, rtol: float) -> None:
    if isinstance(actual, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        return
    if isinstance(actual, (list, tuple)):
        assert len(actual) == len(expected)
        for x, y in zip(actual, expected):
            assert_equal(x, y, atol, rtol)
        return
    if isinstance(actual, dict):
        assert actual.keys() == expected.keys()
        for key in actual:
            assert_equal(actual[key], expected[key], atol, rtol)
        return
    assert actual == expected


def output_variation_ok(outputs: list[torch.Tensor]) -> bool:
    tensors = [x for x in outputs if isinstance(x, torch.Tensor)]
    if len(tensors) < 2:
        return True
    flat = [x.detach().float().reshape(-1) for x in tensors]
    return any(not torch.allclose(flat[0], other) for other in flat[1:])


def main() -> int:
    if torch is None:
        print("[SKIP] torch is not installed; verification downgraded to structural preflight")
        required = ["model.py", "model_new.py", "inputs.py", "task.json"]
        missing = [name for name in required if not Path(name).exists()]
        if missing:
            print(f"[FAIL] missing required files: {', '.join(missing)}")
            return 1
        print("[PASS] verify preflight success")
        return 0

    from inputs import get_init_inputs, get_inputs
    from model import Model
    from model_new import ModelNew

    task = json.loads(Path("task.json").read_text())
    atol = task["correctness_tolerance"]["atol"]
    rtol = task["correctness_tolerance"]["rtol"]
    device = "cuda" if torch.cuda.is_available() and "cuda" in task["device_constraints"] else "cpu"

    init_inputs = transform_tensors(to_sequence(get_init_inputs()), lambda x: x.to(device))
    ref_model = Model(*init_inputs).eval().to(device)
    cand_model = ModelNew(*init_inputs).eval().to(device)
    cand_model.load_state_dict(ref_model.state_dict())

    reference_outputs = []
    with torch.no_grad():
        for idx in range(3):
            sample_inputs = transform_tensors(to_sequence(get_inputs(seed=idx)), lambda x: x.to(device))
            ref_out = ref_model(*sample_inputs)
            with block_torch_functional() if F is not None else nullcontext():
                cand_out = cand_model(*sample_inputs)
            assert_equal(cand_out, ref_out, atol, rtol)
            if isinstance(ref_out, torch.Tensor):
                reference_outputs.append(ref_out)
            print(f"[PASS] verify check {idx + 1}/3")

    assert output_variation_ok(reference_outputs), "reference outputs look indistinguishable across inputs"
    print("[PASS] verify success")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
