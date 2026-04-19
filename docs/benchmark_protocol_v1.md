# Benchmark Protocol V1

Scope:

- Phase-0 baseline protocol for `compile`, `verify`, and `profile`
- Supports eager, optional `torch.compile`, and `model_new.py`

Rules:

- Keep task-defined input shapes fixed for a run.
- Use warmup before timed runs.
- Use `median` as the primary latency metric.
- Record `mean`, `stdev`, and per-iteration samples when possible.
- Separate compile cost from steady-state execution cost.
- Do not run concurrent profiling jobs on the same GPU.
- Treat a result as "strong pass" only when it exceeds the chosen baseline by a stable threshold.

Phase-0 defaults:

- Warmup iterations: 5
- Timed iterations: 20
- Synchronize before and after timed sections when using CUDA
- Compare `eager`, `torch.compile` when available, and `model_new.py`
