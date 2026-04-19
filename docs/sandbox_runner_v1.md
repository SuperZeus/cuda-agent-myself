# Sandbox Runner V1

Sandbox responsibilities:

- Run one action at a time in a task workdir
- Enforce timeout
- Capture stdout and stderr
- Persist a machine-readable event log
- Preserve logs after failure
- Optionally acquire an exclusive profiling lock

Phase-0 guarantees:

- Timeout kills the process group
- Exit status is recorded
- Logs are written even on failures
- Profiling uses a host-level lock file

Deferred to later phases:

- Container isolation
- Cgroup quotas
- Remote GPU pool scheduling
- Multi-host rollout orchestration
