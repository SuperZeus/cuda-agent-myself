You are working inside a standardized CUDA Agent workspace.

Phase-0 constraints:

- `model.py` is the reference implementation for the task.
- `model_new.py` is the candidate implementation.
- `utils/verification.py` and `utils/profiling.py` are protected and should not be modified.
- `compile.py` may succeed without CUDA sources during phase 0.
- When CUDA sources are later added, place them in `kernels/` or the workspace root.

Editable defaults:

- `model_new.py`
- files under `kernels/`

Primary loop:

1. Edit candidate files
2. Run compile
3. Run verify
4. Run profile
5. Iterate
