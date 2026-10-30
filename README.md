# EI-Bench v0.2

A small, single-GPU testbed for an **operational definition of emergent intelligence**
(see [`DEFINITION.md`](DEFINITION.md)). Populations of neural agents invent a discrete
language to describe objects. The harness then scores five pre-registered criteria
(C1–C5) and returns an **EI profile** of PASS / FAIL / INCONCLUSIVE verdicts.

## Quickstart (WSL2 + conda/miniforge)

```bash
bash scripts/setup_env.sh            # creates conda env "eibench", installs, runs env check
conda activate eibench
pytest -q                            # 5 unit tests
python -m eibench.run --quick        # ~2-4 min smoke test -> results/quick
python -m eibench.report --results results/quick
bash scripts/launch.sh 6             # v0.2 grid with 6 parallel workers -> results/v02 (resumable)
python -m eibench.run --progress     # done / running per experiment
python -m eibench.report --results results/v02
```

`--suite v0` reproduces the original grid (into `results/full`). Use `--exp` to pick
experiments, `--dry-run` to list pending runs, and `--device cpu` to force CPU. Any number of
workers can share a grid; runs are claimed with lock files.

## What each experiment does

**v0** (`--suite v0`, 46 runs): c1 (pair × 5 seeds), c3 (N=4 mixed vs. isolated), c5 (hidden
4–256 × 3 seeds), c2 (turnover vs. none, 24k steps).

**v0.2** (default, 160 runs, fresh seeds 100–109):

| Exp | Runs | Steps | Tests |
|---|---|---|---|
| c2b | 3 arms × 10 seeds: reset turnover / matched perturbation / none | 24k | C2 replication, **C2b** perturbation control |
| c3b | het N=4, het N=8, homog N=4 × mixed/isolated × 10 seeds | 8k | **C3b** heterogeneous populations |
| c5b | 12 hidden sizes (4–256) × 5 seeds | 8k | **C5b** transition-width (mirage) test |
| c1 | pair × 10 seeds | 8k | C1/C4 replication |

Populations are implemented with **stacked parameters**: every weight has a leading
agent dimension, and the forward pass uses `torch.baddbmm`. A whole population therefore
trains in one set of batched kernels, and a single agent can be re-initialized
(turnover) by resetting one slice, including its Adam state.

## Layout

```
eibench/   config · data · agents · metrics · stats · train · experiments · run · report
scripts/   setup_env.sh · env_check.py · launch.sh (parallel workers)
tests/     unit tests (split, metrics, reset, perturbation matching, transition width)
results/   created at run time: one JSON per run, REPORT.md, summary.json, figures
```

## Hardware note

PyTorch was chosen over TensorFlow because official TensorFlow wheels do not yet ship
RTX 50-series (sm_120) kernels, while PyTorch has supported Blackwell since 2.7 with
CUDA 12.8. `scripts/env_check.py` verifies that the installed build has kernels for
your GPU.
