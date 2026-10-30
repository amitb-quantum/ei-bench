#!/usr/bin/env bash
# One EI-Bench worker with automatic restart. A crash (e.g. a transient CUDA driver fault)
# restarts the worker after a short pause; finished runs are skipped and its stale lock is
# reclaimed, so nothing is lost or duplicated. Stops after MAX_RESTARTS crashes in a row.
#   usage: scripts/worker.sh <worker-number> [args for eibench.run]
set -u
cd "$(dirname "$0")/.."
ID="$1"; shift
MAX_RESTARTS="${EIBENCH_MAX_RESTARTS:-5}"
for attempt in $(seq 0 "$MAX_RESTARTS"); do
  python -m eibench.run --quiet "$@"
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "[worker $ID] exited cleanly"
    exit 0
  fi
  echo "[worker $ID] crashed with exit code $code (attempt $attempt); restarting in 10 s"
  sleep 10
done
echo "[worker $ID] giving up after $MAX_RESTARTS restarts"
exit 1
