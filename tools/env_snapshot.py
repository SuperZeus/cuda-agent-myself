#!/usr/bin/env python3
"""Collect a lightweight environment snapshot for reproducible runs."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str]) -> str | None:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        output = result.stdout.strip() or result.stderr.strip()
        return output or None
    except Exception:
        return None


def build_snapshot() -> dict:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    snapshot = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": None,
        "cuda_available": None,
        "cuda_version": None,
        "device_count": None,
        "gpu_name": None,
        "nvidia_smi": run_command(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"]),
        "kmp_duplicate_lib_ok": os.environ.get("KMP_DUPLICATE_LIB_OK"),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }

    try:
        import torch

        snapshot["torch_version"] = torch.__version__
        snapshot["cuda_available"] = torch.cuda.is_available()
        snapshot["cuda_version"] = torch.version.cuda
        snapshot["device_count"] = torch.cuda.device_count()
        if torch.cuda.is_available():
            snapshot["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        snapshot["torch_error"] = str(exc)

    return snapshot


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: env_snapshot.py <output-path>")
        return 1
    output_path = Path(sys.argv[1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(build_snapshot(), indent=2) + "\n")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
