"""CLI runner with a file-lock work queue, so several workers can share one GPU.

python -m eibench.run [--suite v0.2] [--exp all|c2b,c3b,...] [--quick] [--device auto]
python -m eibench.run --progress          # done / running / pending per experiment

Any number of workers may run the same command concurrently: each run is claimed with an
exclusive lock file (results/<suite>/.locks/<run_id>.lock); finished runs (JSON present) are
skipped, and locks left by dead processes are reclaimed. Re-running resumes.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from .experiments import DEFAULT_SUITE, SUITES


def default_out(suite: str, quick: bool) -> str:
    name = {"v0": "full", "v0.2": "v02", "v0.3": "v03"}[suite]
    return os.path.join("results", f"{name}_quick" if quick and suite != "v0" else ("quick" if quick else name))


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def claim(lockdir: str, rid: str) -> bool:
    path = os.path.join(lockdir, rid + ".lock")
    for _ in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode())
            os.close(fd)
            return True
        except FileExistsError:
            try:
                pid = int(open(path).read().strip() or 0)
            except (OSError, ValueError):
                pid = 0
            if pid and _pid_alive(pid):
                return False
            try:
                os.remove(path)  # stale lock from a dead worker
            except FileNotFoundError:
                pass
    return False


def release(lockdir: str, rid: str):
    try:
        os.remove(os.path.join(lockdir, rid + ".lock"))
    except FileNotFoundError:
        pass


def progress(cfgs, out, lockdir):
    from collections import Counter
    tot, done, running = Counter(), Counter(), Counter()
    for c in cfgs:
        tot[c.experiment] += 1
        rid = c.run_id()
        if os.path.exists(os.path.join(out, rid + ".json")):
            done[c.experiment] += 1
        elif os.path.exists(os.path.join(lockdir, rid + ".lock")):
            running[c.experiment] += 1
    for e in tot:
        print(f"  {e:5s} {done[e]:3d}/{tot[e]:3d} done, {running[e]} running")
    print(f"  total {sum(done.values())}/{sum(tot.values())} done -> {out}")


def main(argv=None):
    p = argparse.ArgumentParser(description="EI-Bench runner")
    p.add_argument("--suite", default=DEFAULT_SUITE, choices=list(SUITES))
    p.add_argument("--exp", default="all", help="all, or comma-separated experiment names of the suite")
    p.add_argument("--quick", action="store_true", help="tiny smoke-test grid")
    p.add_argument("--device", default="auto")
    p.add_argument("--out", default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--progress", action="store_true")
    p.add_argument("--quiet", action="store_true")
    a = p.parse_args(argv)

    suite = SUITES[a.suite]
    exps = list(suite) if a.exp == "all" else [e.strip() for e in a.exp.split(",")]
    for e in exps:
        if e not in suite:
            sys.exit(f"unknown experiment {e!r} for suite {a.suite}; choose from {list(suite)}")
    out = a.out or default_out(a.suite, a.quick)
    lockdir = os.path.join(out, ".locks")
    os.makedirs(lockdir, exist_ok=True)
    cfgs = [c for e in exps for c in suite[e](quick=a.quick)]

    if a.progress:
        progress(cfgs, out, lockdir)
        return
    pending = [c for c in cfgs if not os.path.exists(os.path.join(out, c.run_id() + ".json"))]
    print(f"[pid {os.getpid()}] suite {a.suite}: {len(cfgs)} runs, {len(pending)} not finished -> {out}", flush=True)
    if a.dry_run:
        for c in pending:
            print("  ", c.run_id())
        return

    import torch
    from .train import save_result, train_run
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.set_num_threads(max(1, int(os.environ.get("EIBENCH_THREADS", "2"))))

    t0, n = time.time(), 0
    for c in pending:
        rid = c.run_id()
        if os.path.exists(os.path.join(out, rid + ".json")) or not claim(lockdir, rid):
            continue
        try:
            if os.path.exists(os.path.join(out, rid + ".json")):
                continue
            print(f"=== {rid} ===", flush=True)
            res = train_run(c, device=a.device, verbose=not a.quiet)
            path = save_result(res, out)
            f = res["final"]
            n += 1
            print(f"    done in {res['wall_s']:.0f}s on {res['device_name']}: train={f['train_acc']:.3f} "
                  f"test={f['test_acc']:.3f} topsim={f['structure']['topsim']:.3f} -> {path}", flush=True)
        finally:
            release(lockdir, rid)
    print(f"[pid {os.getpid()}] finished {n} runs in {(time.time() - t0) / 60:.1f} min. "
          f"Report: python -m eibench.report --results {out}", flush=True)


if __name__ == "__main__":
    main()
