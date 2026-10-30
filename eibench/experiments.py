"""Experiment grids. Suites: "v0" (original, 5 seeds) and "v0.2" (pre-registered follow-ups,
10 fresh seeds). `quick=True` gives a tiny smoke-test version of either suite."""
from __future__ import annotations

from .config import Config


def _base(quick: bool, **kw) -> dict:
    d = dict(steps=8000, eval_every=500)
    if quick:
        d.update(steps=300, eval_every=100)
    d.update(kw)
    return d


def seeds(quick: bool, full: int = 5, offset: int = 0) -> list[int]:
    return [offset, offset + 1] if quick else list(range(offset, offset + full))


def _c2_params(quick):
    # steps, turnover_start, turnover_every, eval_every
    return (900, 300, 100, 100) if quick else (24000, 8000, 500, 1000)


# ------------------------------------------------------------------ v0
def c1(quick=False, n_seeds=5, offset=0):
    """C1 + C4: single sender-receiver pair; held-out generalisation and language structure."""
    return [Config(**_base(quick), pop_size=1, seed=s, experiment="c1", condition="pair")
            for s in seeds(quick, n_seeds, offset)]


def c3(quick=False):
    """C3: mixed vs isolated population (N=4), identical parameters and compute."""
    return [Config(**_base(quick), pop_size=4, mixing=mixing, seed=s, experiment="c3", condition=mixing)
            for mixing in ("mixed", "isolated") for s in seeds(quick)]


def _c2_arm(quick, cond, mode, every, s, experiment):
    steps, start, t_every, ev = _c2_params(quick)
    return Config(steps=steps, eval_every=ev, pop_size=4, mixing="mixed",
                  turnover_every=t_every if every else 0, turnover_start=start, turnover_stop_frac=0.85,
                  turnover_mode=mode, lr_decay_frac=0.15, seed=s, experiment=experiment, condition=cond)


def c2(quick=False):
    """C2: mixed population with turnover (iterated learning) vs. same training without it."""
    return [_c2_arm(quick, cond, "reset", every, s, "c2")
            for cond, every in (("turnover", True), ("no_turnover", False)) for s in seeds(quick)]


def c5(quick=False):
    """C5: scale sweep over hidden size, single pair."""
    sizes = [8, 32, 128] if quick else [4, 8, 16, 32, 64, 128, 256]
    ss = [0, 1] if quick else [0, 1, 2]
    return [Config(**_base(quick), pop_size=1, hidden=h, embed=min(32, max(8, h // 2)), seed=s,
                   experiment="c5", condition=f"h{h}", tags={"hidden": h})
            for h in sizes for s in ss]


# ------------------------------------------------------------------ v0.2
V02_SEEDS = 100  # fresh seeds 100..109, independent of v0


def c1_rep(quick=False):
    return c1(quick, n_seeds=10, offset=V02_SEEDS)


def c2b(quick=False):
    """C2-rep + C2b: reset turnover vs displacement-matched perturbation vs none."""
    arms = (("turnover", "reset", True), ("perturb", "perturb", True), ("no_turnover", "reset", False))
    return [_c2_arm(quick, cond, mode, every, s, "c2b")
            for cond, mode, every in arms for s in seeds(quick, 10, V02_SEEDS)]


def c3b(quick=False):
    """C3b: heterogeneous-learning-rate populations, mixed vs isolated; N=8 and homogeneous as secondary."""
    arms = []
    for tag, n, spread in (("het", 4, 2.0), ("het8", 8, 2.0), ("homog", 4, 1.0)):
        for mixing in ("mixed", "isolated"):
            arms.append((f"{tag}_{mixing}", n, spread, mixing))
    return [Config(**_base(quick), pop_size=n, mixing=mixing, lr_spread=spread, seed=s,
                   experiment="c3b", condition=cond, tags={"group": cond.rsplit("_", 1)[0]})
            for cond, n, spread, mixing in arms for s in seeds(quick, 10, V02_SEEDS)]


C5B_SIZES = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def c5b(quick=False):
    """C5b: fine scale sweep for the transition-width (mirage) test."""
    sizes = [8, 16, 32, 64, 128] if quick else C5B_SIZES
    return [Config(**_base(quick), pop_size=1, hidden=h, embed=min(32, max(8, h // 2)), seed=s,
                   experiment="c5b", condition=f"h{h}", tags={"hidden": h})
            for h in sizes for s in seeds(quick, 5, V02_SEEDS)]


SUITES = {
    "v0": {"c1": c1, "c3": c3, "c5": c5, "c2": c2},
    # ordered longest-first so parallel workers finish together
    "v0.2": {"c2b": c2b, "c3b": c3b, "c5b": c5b, "c1": c1_rep},
}
DEFAULT_SUITE = "v0.2"
