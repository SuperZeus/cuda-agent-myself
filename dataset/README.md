# Dataset Pipeline

This directory stores phase-3 dataset artifacts and split manifests.

Key subdirectories:

- `raw/`: optional imported sources
- `processed/`: generated task manifests and filter reports
- `splits/`: frozen split files
- `trajectories/`: batch rollout and replay summaries

Primary commands:

```bash
python3 tools/build_phase3_dataset.py
python3 tools/sample_dataset_trajectories.py --limit 12
python3 tools/validate_phase3_dataset.py
```
