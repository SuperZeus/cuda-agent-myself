# Environment Contract

Phase-0 environment policy:

- `configs/env.local.example.json` documents the local development contract.
- `configs/env.standard.json` documents the intended locked evaluation contract.
- Every task run records an environment snapshot to `env_snapshot.json`.

Minimum environment fields captured:

- Python version
- Platform and machine info
- Torch version
- CUDA availability
- CUDA version
- GPU name
- `nvidia-smi` query output when available
- OpenMP compatibility overrides used by the sandbox

This is enough for phase 0 reproducibility and can be extended in later phases.
