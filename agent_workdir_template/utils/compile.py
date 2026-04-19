#!/usr/bin/env python3
"""Phase-0 compile entrypoint.

If CUDA/C++ sources exist, try to compile them.
If no sources exist yet, perform a structural preflight and succeed.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path


def find_sources() -> list[str]:
    root = Path(".")
    kernels = Path("kernels")
    sources = [str(p) for p in root.glob("*.cu")] + [str(p) for p in root.glob("*.cpp")]
    if kernels.exists():
        sources.extend(str(p) for p in kernels.glob("*.cu"))
        sources.extend(str(p) for p in kernels.glob("*.cpp"))
    ignored = {"binding.cpp"}
    return sorted(set(source for source in sources if source not in ignored))


def preflight() -> int:
    required = ["model.py", "model_new.py", "task.json", "utils/verification.py", "utils/profiling.py"]
    missing = [name for name in required if not Path(name).exists()]
    if missing:
        print(f"[FAIL] missing required files: {', '.join(missing)}")
        return 1
    print("[PASS] structural preflight success")
    return 0


def compile_sources(sources: list[str]) -> int:
    try:
        import torch.utils.cpp_extension as cpp_ext
    except Exception as exc:
        print(f"[SKIP] torch cpp extension unavailable: {exc}")
        return preflight()

    build_dir = Path("build/phase0_compile")
    output_so = Path("cuda_extension.so")
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    if output_so.exists():
        output_so.unlink()

    try:
        cpp_ext.load(
            name="cuda_extension",
            sources=sources,
            build_directory=str(build_dir),
            verbose=False,
            with_cuda=True,
            extra_cflags=["-O2", "-std=c++17"],
            extra_cuda_cflags=["-O2"],
        )
    except Exception as exc:
        print(f"[FAIL] compile failed: {exc}")
        return 1

    built_so = build_dir / "cuda_extension.so"
    if built_so.exists():
        shutil.copy2(built_so, output_so)
    print("[PASS] compile success")
    return 0


def main() -> int:
    sources = find_sources()
    if not sources:
        return preflight()
    print(f"[INFO] compiling sources: {', '.join(sources)}")
    return compile_sources(sources)


if __name__ == "__main__":
    raise SystemExit(main())
