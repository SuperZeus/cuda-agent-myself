# cuda-agent-myself

This repository implements the phase-0 foundation for a CUDA Agent system based on [cuda_agent_plan_v3.md](/Users/liumingliang/CUDA/cuda_agent_plan_v3.md).

Current scope:

- Repository skeleton aligned with the V3 architecture
- Standard task schema and workspace metadata schema
- Standard agent workdir template
- Sandbox runner v1 with timeout, logging, and profiling lock support
- Observation compression and read-only evaluation policies
- Ten standardized starter tasks
- One-command `compile` / `verify` / `profile` task execution
- Phase-2 inference-time agent MVP with persistent episode trajectories
- Phase-3 dataset generation, split freezing, and episode-level rollout sampling

Quick start:

```bash
cd /Users/liumingliang/CUDA/cuda-agent-myself
python3 -m runner.run_task --task-id axpby_scalar --action compile
python3 -m runner.run_task --task-id axpby_scalar --action verify
python3 -m runner.run_task --task-id axpby_scalar --action profile
python3 -m runner.run_task --task-id axpby_scalar --variant handcrafted --action verify
python3 tools/validate_tasks.py
python3 tools/phase0_smoke.py --actions compile verify
python3 tools/phase1_report.py
python3 tools/run_agent_mvp.py --task-id axpby_scalar --policy handcrafted_bootstrap
python3 tools/run_agent_mvp.py --task-id softmax_rows --policy agent_edit_loop_v1
python3 tools/replay_viewer.py /path/to/agent_trajectory.json
python3 tools/phase2_report.py
python3 tools/build_phase3_dataset.py
python3 tools/sample_dataset_trajectories.py --limit 10 --rounds 4
python3 tools/validate_phase3_dataset.py
```

Local environment note:

- The sandbox sets `KMP_DUPLICATE_LIB_OK=TRUE` and `OMP_NUM_THREADS=1` for child processes.
- This avoids a known duplicate OpenMP runtime crash in some local Torch builds.

Phase 0 intentionally keeps compilation permissive:

- If CUDA/C++ sources exist, `compile.py` attempts extension compilation.
- If no CUDA/C++ sources exist yet, `compile.py` performs a structural preflight and succeeds.

That keeps the workflow runnable before phase 1 introduces real CUDA kernels.
