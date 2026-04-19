# Phase 2 Agent MVP

This repository now includes a phase-2 inference-time agent runner.

Capabilities:

- standardized episode instantiation
- deterministic observation construction
- tool-like steps for `list_files`, `read_file`, bootstrap writes, and `compile/verify/profile`
- success and termination logic
- trajectory and summary persistence
- replay viewing

Primary commands:

```bash
python3 tools/run_agent_mvp.py --task-id axpby_scalar --policy handcrafted_bootstrap
python3 tools/replay_viewer.py /path/to/agent_trajectory.json
```

Available policies:

- `baseline_reference`
- `reference_full_loop`
- `handcrafted_bootstrap`
- `handcrafted_full_loop`
- `agent_edit_loop_v1`

`agent_edit_loop_v1` edits `model_new.py` inside the instantiated workspace, runs
`compile/verify/profile`, and performs one extra rewrite when profile feedback
shows weak eager speedup. If the tuned rewrite breaks correctness, it restores a
safe explicit candidate and retries verification.

Phase-3 note:

`tools/sample_dataset_trajectories.py` should prefer these episode-level traces over flat action manifests.
