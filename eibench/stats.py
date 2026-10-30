"""Bootstrap CIs and verdict logic (seeds are the unit of replication)."""
from __future__ import annotations

import numpy as np

N_BOOT = 10_000


def ci_mean(x, alpha=0.05, seed=0):
    x = np.asarray(x, float)
    rng = np.random.default_rng(seed)
    bs = rng.choice(x, size=(N_BOOT, len(x)), replace=True).mean(1)
    return float(x.mean()), float(np.quantile(bs, alpha / 2)), float(np.quantile(bs, 1 - alpha / 2))


def ci_diff(a, b, alpha=0.05, seed=0):
    """Unpaired difference of means, mean(a) - mean(b)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    ba = rng.choice(a, size=(N_BOOT, len(a)), replace=True).mean(1)
    bb = rng.choice(b, size=(N_BOOT, len(b)), replace=True).mean(1)
    d = ba - bb
    return float(a.mean() - b.mean()), float(np.quantile(d, alpha / 2)), float(np.quantile(d, 1 - alpha / 2))


def verdict_above(ci, threshold=0.0):
    """PASS if the whole CI is above threshold; INCONCLUSIVE if only the mean is; else FAIL."""
    mean, lo, _ = ci
    if lo > threshold:
        return "PASS"
    if mean > threshold:
        return "INCONCLUSIVE"
    return "FAIL"


def jump_ratio(values) -> float:
    """Largest single-step rise of a curve divided by its total range (0..1)."""
    v = np.asarray(values, float)
    rng = v.max() - v.min()
    if len(v) < 2 or rng <= 1e-9:
        return 0.0
    return float(max(np.diff(v).max(), 0.0) / rng)
