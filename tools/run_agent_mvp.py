#!/usr/bin/env python3
"""Run the phase-2 inference-time agent MVP for a task."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runner.agent_episode import POLICIES, run_episode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--policy", choices=sorted(POLICIES), default="baseline_reference")
    args = parser.parse_args()

    episode = run_episode(args.task_id, args.policy)
    print(json.dumps(episode, indent=2))
    return 0 if episode["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
