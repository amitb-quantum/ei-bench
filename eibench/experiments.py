"""Experiment grids for each criterion. `quick=True` gives a smoke-test version."""
from __future__ import annotations

from .config import Config


def _base(quick: bool, **kw) -> dict:
    d = dict(steps=8000, eval_every=500)
    if quick:
        d.update(steps=300, eval_every=100)
    d.update(kw)
    return d


def seeds(quick: bool, full: int = 5) -> list[int]:
    return [0, 1] if quick else list(range(full))


def c1(quick=False):
    """C1 + C4: single sender-receiver pair; held-out generalisation and language structure."""
    return [Config(**_base(quick), pop_size=1, seed=s, experiment="c1", condition="pair")
            for s in seeds(quick)]


def c3(quick=False):
    """C3: mixed vs isolated population (N=4), identical parameters and compute."""
    out = []
    for mixing in ("mixed", "isolated"):
        for s in seeds(quick):
            out.append(Config(**_base(quick), pop_size=4, mixing=mixing, seed=s,
                              experiment="c3", condition=mixing))
    return out


def c2(quick=False):
    """C2: mixed population with turnover (iterated learning) vs. same training without it."""
    if quick:
        steps, start, every, ev = 900, 300, 100, 100
    else:
        steps, start, every, ev = 24000, 8000, 500, 1000
    out = []
    for cond, t_every in (("turnover", every), ("no_turnover", 0)):
        for s in seeds(quick):
            out.append(Config(steps=steps, eval_every=ev, pop_size=4, mixing="mixed",
                              turnover_every=t_every, turnover_start=start, turnover_stop_frac=0.85,
                              lr_decay_frac=0.15,  # decay only during the settle phase
                              seed=s, experiment="c2", condition=cond))
    return out


def c5(quick=False):
    """C5: scale sweep over hidden size, single pair."""
    sizes = [8, 32, 128] if quick else [4, 8, 16, 32, 64, 128, 256]
    ss = [0, 1] if quick else [0, 1, 2]
    return [Config(**_base(quick), pop_size=1, hidden=h, embed=min(32, max(8, h // 2)), seed=s,
                   experiment="c5", condition=f"h{h}", tags={"hidden": h})
            for h in sizes for s in ss]


EXPERIMENTS = {"c1": c1, "c2": c2, "c3": c3, "c5": c5}
ORDER = ["c1", "c3", "c5", "c2"]  # cheapest first; c2 is the long one
