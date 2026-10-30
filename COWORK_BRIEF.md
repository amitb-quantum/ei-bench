# Brief for Claude (Cowork): run EI-Bench v0.2 on vermithor

**Owner:** AB (Quantum Clarity LLC). **Machine:** `vermithor`, WSL2 Ubuntu 26.04, RTX 5090, conda env `eibench`.
**State:** v0 is installed at `~/projects/ei-bench` (local git repo; v0 results in `results/full`).
**Goal:** upgrade the code to v0.2, verify it, run the pre-registered v0.2 grid, and hand back the results.
Read `DEFINITION.md` §6–7 first: it lists the new tests and their pass thresholds.

## Ground rules

1. **Preservation first.** Do not touch `results/full` (v0 results), other conda envs, or anything
   outside `~/projects/ei-bench`. v0.2 writes to `results/v02`.
2. **Do not change the science.** The thresholds in `DEFINITION.md` and the grids in
   `eibench/experiments.py` are pre-registered. For a real bug: fix it, log it in `CHANGES.md`,
   delete only the affected JSONs, and re-run them. Never tune to change a verdict.
3. Local git only, no pushing. Stop and ask AB if a step fails twice the same way.

## Steps

### 1. Upgrade in place
The archive `ei-bench-v0.2.tar.gz` is in `/tmp` (or AB's Windows Downloads folder).
```bash
cd ~/projects/ei-bench
git status --short                      # should be clean; if not, commit first
tar xzf /tmp/ei-bench-v0.2.tar.gz -C ~/projects   # overwrites code only; results/ is not in the archive
conda activate eibench
git add -A && git commit -qm "EI-Bench v0.2 code + pre-registration"
pytest -q                               # expect: 8 passed
```

### 2. Smoke test (about 3–5 min)
```bash
bash scripts/launch.sh 3 --quick
python -m eibench.run --progress --quick      # repeat until 30/30
python -m eibench.report --results results/v02_quick
```
**Checkpoint:** 30/30 runs, no `Traceback` in `logs/worker_*.log`, and the report lists C1, C2, C2b, C3b, C4
and C5b. The quick verdicts are meaningless (300 steps). Clear the old logs before the full run: `rm logs/*.log`.

### 3. Full v0.2 grid: 160 runs, expect about 2–3 h
```bash
bash scripts/launch.sh 6
watch -n 60 python -m eibench.run --progress     # or check it now and then
nvidia-smi                                         # GPU should be busy
```
Six workers share one work queue (lock files), and the longest runs are scheduled first. If anything
dies, run `bash scripts/launch.sh 6` again; finished runs are skipped and stale locks are reclaimed.
If the GPU looks saturated (~100%) and runs are slowing down, 4 workers is fine as well.

Anomalies to record (do not re-run or drop them): `nan` in any log, and any c1/c3b run with final
`train` below 0.9.

### 4. Report and package
```bash
python -m eibench.report --results results/v02
tar czf ei-bench-results-v02.tar.gz results/v02/REPORT.md results/v02/summary.json results/v02/*.png logs/
cp ei-bench-results-v02.tar.gz /mnt/c/Users/<windows-user>/Downloads/ 2>/dev/null || true
git add -A && git commit -qm "v0.2 results" || true
```

## What to report back to AB
1. The EI-profile table from `results/v02/REPORT.md`, verbatim, plus the per-seed lines for C2/C2b.
2. Wall time, number of workers used, and GPU utilization seen.
3. Anomalies and any `CHANGES.md` entries.
4. Where `ei-bench-results-v02.tar.gz` is.
