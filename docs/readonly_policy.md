# Read-Only Evaluation Policy

The following files are treated as protected by default when a task is instantiated:

- `utils/verification.py`
- `utils/profiling.py`
- `task.json`
- `workspace_meta.json`

Policy goals:

- Prevent reward hacking by modifying evaluation scripts
- Keep benchmark semantics stable across runs
- Preserve task metadata and run metadata integrity

Phase-0 enforcement:

- The runner marks protected files read-only with filesystem permissions.
- Protected files are also listed in `workspace_meta.json`.
