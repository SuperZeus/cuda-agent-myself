#!/usr/bin/env python3
"""Run an action inside a task workdir with timeout, logging, and optional profiling lock."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path


@contextmanager
def profiling_lock(enabled: bool, lock_path: Path):
    if not enabled:
        yield
        return

    import fcntl

    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--action", required=True)
    parser.add_argument("--timeout-sec", type=int, required=True)
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--profile-lock", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser.parse_args()


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def main() -> int:
    args = parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        print("missing command")
        return 2

    log_dir = Path(args.log_dir)
    stdout_path = log_dir / f"{args.action}.stdout.log"
    stderr_path = log_dir / f"{args.action}.stderr.log"
    event_path = log_dir / f"{args.action}.event.json"

    started_at = datetime.now(timezone.utc).isoformat()
    start = time.time()
    timed_out = False

    child_env = os.environ.copy()
    # Local Torch builds on macOS can load duplicate OpenMP runtimes.
    child_env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    child_env.setdefault("OMP_NUM_THREADS", "1")

    with profiling_lock(args.profile_lock, log_dir / ".profiling.lock"):
        process = subprocess.Popen(
            args.command,
            cwd=args.workdir,
            env=child_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid,
        )
        try:
            stdout, stderr = process.communicate(timeout=args.timeout_sec)
            exit_code = process.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            exit_code = 124

    finished_at = datetime.now(timezone.utc).isoformat()
    write_text(stdout_path, stdout)
    write_text(stderr_path, stderr)
    event = {
        "run_id": args.run_id,
        "task_id": args.task_id,
        "action": args.action,
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_sec": round(time.time() - start, 3),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "command": args.command,
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
    }
    event_path.write_text(json.dumps(event, indent=2) + "\n")
    print(event_path)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
