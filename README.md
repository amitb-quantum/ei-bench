# EI-Bench v0

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
python -m eibench.run                # full grid -> results/full (resumable)
python -m eibench.report --results results/full
```

Run a single criterion with `--exp c1` (or `c2`, `c3`, `c5`; C4 is scored from the
c1 runs). `--dry-run` lists pending runs, and `--device cpu` forces CPU.

## What each experiment does

| Exp | Runs (full) | Steps | Tests |
|---|---|---|---|
| c1 | 1 pair × 5 seeds | 8k | C1 held-out generalization, C4 structure vs. untrained |
| c3 | N=4 mixed vs. isolated × 5 seeds | 8k | C3 interaction effect at equal compute |
| c5 | hidden ∈ {4…256} × 3 seeds | 8k | C5 jump vs. continuous metrics |
| c2 | N=4 turnover vs. none × 5 seeds | 24k | C2 ongoing development |

Populations are implemented with **stacked parameters**: every weight has a leading
agent dimension, and the forward pass uses `torch.baddbmm`. A whole population therefore
trains in one set of batched kernels, and a single agent can be re-initialized
(turnover) by resetting one slice, including its Adam state.

## Layout

```
eibench/   config · data · agents · metrics · stats · train · experiments · run · report
scripts/   setup_env.sh · env_check.py
tests/     unit tests (split coverage, metric sanity, population reset)
results/   created at run time: one JSON per run, REPORT.md, summary.json, figures
```

## Hardware note

PyTorch was chosen over TensorFlow because official TensorFlow wheels do not yet ship
RTX 50-series (sm_120) kernels, while PyTorch has supported Blackwell since 2.7 with
CUDA 12.8. `scripts/env_check.py` verifies that the installed build has kernels for
your GPU.
