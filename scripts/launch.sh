#!/usr/bin/env bash
# Start N self-restarting workers that share one work queue on one GPU.
#   bash scripts/launch.sh            # 6 workers, default suite (v0.3)
#   bash scripts/launch.sh 4 --quick  # 4 workers, smoke-test grid
# Extra args are passed to `python -m eibench.run`. Safe to re-run: finished runs are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-6}"; shift || true
mkdir -p logs
export EIBENCH_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
for i in $(seq 1 "$N"); do
  nohup bash scripts/worker.sh "$i" "$@" > "logs/worker_${i}.log" 2>&1 &
  echo "worker $i -> pid $! (logs/worker_${i}.log)"
  sleep 2   # stagger start-up so workers claim different runs
done
echo
echo "progress:  python -m eibench.run --progress $*"
echo "logs:      tail -f logs/worker_*.log"
echo "restarts:  grep -h crashed logs/worker_*.log"
echo "stop all:  pkill -f scripts/worker.sh; pkill -f eibench.run"
