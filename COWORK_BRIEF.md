# Brief for Claude (Cowork) — run EI-Bench v0 on vermithor

**Owner:** AB (Quantum Clarity LLC). **Machine:** `vermithor`, WSL2 Ubuntu 26.04, RTX 5090, miniforge3 conda.
**Goal:** set up EI-Bench v0, verify it, run the full experiment grid, and hand back the results.
Read `README.md` and `DEFINITION.md` first.

## Ground rules

1. **Preservation first.** Create only the new conda env `eibench`. Do not modify, update, or
   remove any existing conda env, system package, or file outside `~/projects/ei-bench`.
2. **Do not change the science.** The thresholds in `DEFINITION.md` and the grids in
   `eibench/experiments.py` are pre-registered. If you find a real bug, fix it, record it in
   `CHANGES.md` (what, why, which runs it affects), delete only the affected result JSONs, and
   re-run them. Never tune hyperparameters to change a verdict.
3. **No pushing anywhere.** Use a local git repo only.
4. Stop and ask AB if a step fails twice in the same way.

## Steps

### 0. Unpack
The archive `ei-bench-v0.1.tar.gz` is in AB's Windows Downloads folder
(`/mnt/c/Users/<windows-user>/Downloads/`). Find it with
`ls /mnt/c/Users/*/Downloads/ei-bench-v0.1.tar.gz`, then:
```bash
mkdir -p ~/projects && tar xzf /mnt/c/Users/<windows-user>/Downloads/ei-bench-v0.1.tar.gz -C ~/projects
cd ~/projects/ei-bench
```
If `~/projects/ei-bench` already exists, stop and ask before overwriting.

### 1. Environment
```bash
nvidia-smi                                  # GPU visible inside WSL?
source ~/miniforge3/bin/activate            # if conda is not already on PATH
bash scripts/setup_env.sh                   # ends by running scripts/env_check.py
```
**Checkpoint:** `env_check.py` prints `READY`, shows the RTX 5090 with compute capability 12.0,
and lists `sm_120` among the compiled archs. If CUDA is not available, check `nvidia-smi` and the
Windows NVIDIA driver; do not install a CUDA toolkit inside WSL. If it still fails, report it
and continue on CPU only with AB's OK.

### 2. Tests and smoke run
```bash
conda activate eibench
pytest -q                                           # expect: 5 passed
python -m eibench.run --quick --quiet
python -m eibench.report --results results/quick
```
**Checkpoint:** 16 quick runs complete and the report is written. The quick-mode verdicts are
meaningless (300 steps); this step only proves the pipeline works on the GPU. Note the
per-run times. The `device_name` in the JSONs should be the 5090.

### 3. Snapshot
```bash
git init -q && git add -A && git commit -qm "EI-Bench v0.1 as received"
```

### 4. Full run (expected about 1–2 h on the 5090; resumable)
```bash
nohup python -m eibench.run > results_full.log 2>&1 &
tail -f results_full.log
```
Order: c1 → c3 → c5 → c2 (c2 is the long one). If the process dies, run the same command again;
finished runs are skipped. While it runs, watch for:
- `nan` in any logged metric,
- any c1 or c3 run finishing with `train` accuracy below 0.9 (a non-converged run).

Record these as anomalies; do not re-run or drop them.

### 5. Report and package
```bash
python -m eibench.report --results results/full
tar czf ei-bench-results-full.tar.gz results/full/REPORT.md results/full/summary.json \
    results/full/*.png results_full.log CHANGES.md 2>/dev/null || \
tar czf ei-bench-results-full.tar.gz results/full/REPORT.md results/full/summary.json results/full/*.png results_full.log
cp ei-bench-results-full.tar.gz /mnt/c/Users/<windows-user>/Downloads/
git add -A && git commit -qm "Full v0 results" || true
```

## What to report back to AB
1. The EI-profile table from `results/full/REPORT.md`, verbatim.
2. Total wall time and the GPU used.
3. Anomalies (NaNs, non-converged runs, any bug fixes listed in `CHANGES.md`).
4. Confirmation that `ei-bench-results-full.tar.gz` is in Downloads, so AB can bring it back
   to the planning chat.
