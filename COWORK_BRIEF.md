# Brief for Claude (Cowork): run EI-Bench v0.3 on vermithor

**Owner:** AB (Quantum Clarity LLC). **Machine:** `vermithor`, WSL2 Ubuntu 26.04, RTX 5090, conda env `eibench`.
**State:** EI-Bench is installed at `~/projects/ei-bench` (git repo pushed to amitb-quantum/ei-bench; v0 results in `results/full`, v0.2 in `results/v02`).
**Goal:** upgrade the code to v0.3, verify it, run the pre-registered v0.3 grid, and hand back the results.
Read `DEFINITION.md` §9 first: it lists the new tests and their verdict rules.

## Ground rules

1. **Preservation first.** Do not touch `results/full` or `results/v02`, other conda envs, or anything
   outside `~/projects/ei-bench`. v0.3 writes to `results/v03`.
2. **Do not change the science.** The thresholds in `DEFINITION.md` and the grids in
   `eibench/experiments.py` are pre-registered. For a real bug: fix it, log it in `CHANGES.md`,
   delete only the affected JSONs, and re-run them. Never tune to change a verdict.
3. Local git only, no pushing. Stop and ask AB if a step fails twice the same way.

## Steps

### 1. Upgrade in place
The archive `ei-bench-v0.3.tar.gz` is in `/tmp` (or AB's Windows Downloads folder).
```bash
cd ~/projects/ei-bench
git status --short                      # should be clean; if not, commit first
tar xzf /tmp/ei-bench-v0.3.tar.gz -C ~/projects   # overwrites code only; results/ is not in the archive
conda activate eibench
git add -A && git commit -qm "EI-Bench v0.3 code + pre-registration" && git push
pytest -q                               # expect: 10 passed
```

### 2. Smoke test (a few minutes)
```bash
bash scripts/launch.sh 3 --quick
python -m eibench.run --progress --quick      # repeat until 8/8
python -m eibench.report --results results/v03_quick
```
**Checkpoint:** 8/8 runs, no `Traceback` in `logs/worker_*.log`, and the report lists C1, C2, C2c and C4. The quick verdicts are meaningless (300 steps). Clear the old logs before the full run: `rm logs/*.log`.

### 3. Full v0.3 grid: 50 runs, expect about 1.5–2 h
```bash
bash scripts/launch.sh 6
watch -n 60 python -m eibench.run --progress     # or check it now and then
nvidia-smi                                         # GPU should be busy
```
Six self-restarting workers share one work queue (lock files); the longest runs are scheduled first.
A crashed worker restarts itself (`grep -h crashed logs/worker_*.log`). If everything stops, run
`bash scripts/launch.sh 6` again; finished runs are skipped and stale locks are reclaimed.

Anomalies to record (do not re-run or drop them): `nan` in any log, any worker restarts, and any c1
run with final `train` below 0.9.

### 4. Report and package
```bash
python -m eibench.report --results results/v03
git add -f results/v03/REPORT.md results/v03/summary.json results/v03/*.png
git commit -qm "v0.3 results" && git push
```

## What to report back to AB
1. `results/v03/REPORT.md`, verbatim.
2. Wall time, number of workers used, and GPU utilization seen.
3. Anomalies and any `CHANGES.md` entries.
4. Confirmation that the results commit was pushed.
