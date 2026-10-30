"""CLI: python -m eibench.run --exp all [--quick] [--device auto|cuda|cpu] [--out results]

Runs are resumable: a run whose JSON already exists in --out is skipped.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch

from .experiments import EXPERIMENTS, ORDER
from .train import save_result, train_run


def main(argv=None):
    p = argparse.ArgumentParser(description="EI-Bench v0 runner")
    p.add_argument("--exp", default="all", help="all | c1,c2,c3,c5 (C4 is computed from c1 runs)")
    p.add_argument("--quick", action="store_true", help="tiny smoke-test grid")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None, help="results dir (default results/full or results/quick)")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    exps = ORDER if a.exp == "all" else [e.strip() for e in a.exp.split(",")]
    for e in exps:
        if e not in EXPERIMENTS:
            sys.exit(f"unknown experiment {e!r}; choose from {list(EXPERIMENTS)}")
    out = a.out or os.path.join("results", "quick" if a.quick else "full")

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    cfgs = [c for e in exps for c in EXPERIMENTS[e](quick=a.quick)]
    todo = [c for c in cfgs if not os.path.exists(os.path.join(out, c.run_id() + ".json"))]
    print(f"{len(cfgs)} runs in grid, {len(cfgs) - len(todo)} already done, {len(todo)} to run -> {out}")
    if a.dry_run:
        for c in todo:
            print("  ", c.run_id())
        return

    t0 = time.time()
    for i, c in enumerate(todo, 1):
        print(f"=== [{i}/{len(todo)}] {c.run_id()} ===", flush=True)
        res = train_run(c, device=a.device, verbose=not a.quiet)
        path = save_result(res, out)
        f = res["final"]
        print(f"    done in {res['wall_s']:.0f}s on {res['device_name']}: "
              f"test_acc={f['test_acc']:.3f} topsim={f['structure']['topsim']:.3f} -> {path}", flush=True)
    print(f"all done in {(time.time() - t0) / 60:.1f} min. Next: python -m eibench.report --results {out}")


if __name__ == "__main__":
    main()
