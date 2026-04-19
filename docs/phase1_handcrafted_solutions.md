# Phase 1 Handcrafted CUDA Solutions

This repository now includes handcrafted CUDA-oriented variants for the following tasks:

- `axpby_scalar`
- `broadcast_add`
- `relu_bias`
- `sigmoid_mul`
- `sum_reduction_lastdim`
- `mean_reduction_lastdim`

Each handcrafted variant is stored under:

```text
tasks/<task_id>/handcrafted/
├── model_new.py
└── kernels/
```

Design notes:

- `model_new.py` prefers a compiled `cuda_extension` path when available.
- Each handcrafted solution also has a Torch fallback so verification and profiling still run in non-CUDA environments.
- Kernel bindings use the standardized `binding.cpp` / `binding_registry.h` infrastructure from the workdir template.

How to run a handcrafted variant:

```bash
python3 -m runner.run_task --task-id axpby_scalar --variant handcrafted --action compile
python3 -m runner.run_task --task-id axpby_scalar --variant handcrafted --action verify
python3 -m runner.run_task --task-id axpby_scalar --variant handcrafted --action profile
```

Batch summary:

```bash
python3 tools/phase1_report.py
```
