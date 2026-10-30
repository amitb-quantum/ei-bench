"""Structure metrics for emergent languages (numpy only).

* topsim  - Spearman correlation between pairwise meaning distance (Hamming over
            attributes) and message distance (Hamming over fixed-length messages).
* posdis  - positional disentanglement (Chaabouni et al. 2020).
* bosdis  - bag-of-symbols disentanglement (Chaabouni et al. 2020).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def topsim(meanings: np.ndarray, messages: np.ndarray, max_n: int = 1000, seed: int = 0) -> float:
    rng = np.random.default_rng(seed)
    n = len(meanings)
    if n > max_n:
        idx = rng.choice(n, max_n, replace=False)
        meanings, messages = meanings[idx], messages[idx]
        n = max_n
    iu = np.triu_indices(n, k=1)
    dm = (meanings[:, None, :] != meanings[None, :, :]).sum(-1)[iu]
    ds = (messages[:, None, :] != messages[None, :, :]).sum(-1)[iu]
    if ds.std() == 0 or dm.std() == 0:
        return 0.0
    return float(spearmanr(dm, ds).statistic)


def _entropy(x: np.ndarray) -> float:
    _, counts = np.unique(x, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def _mutual_info(x: np.ndarray, y: np.ndarray) -> float:
    joint = x.astype(np.int64) * (int(y.max()) + 1) + y.astype(np.int64)
    return _entropy(x) + _entropy(y) - _entropy(joint)


def _disentanglement(features: np.ndarray, meanings: np.ndarray) -> float:
    """features: (n, F) discrete variables; meanings: (n, A). Requires A >= 2."""
    scores = []
    A = meanings.shape[1]
    for j in range(features.shape[1]):
        f = features[:, j]
        h = _entropy(f)
        if h <= 1e-12:
            continue
        mi = sorted((_mutual_info(f, meanings[:, a]) for a in range(A)), reverse=True)
        scores.append((mi[0] - mi[1]) / h)
    return float(np.mean(scores)) if scores else 0.0


def posdis(meanings: np.ndarray, messages: np.ndarray) -> float:
    return _disentanglement(messages, meanings)


def bosdis(meanings: np.ndarray, messages: np.ndarray, vocab_size: int) -> float:
    counts = np.stack([(messages == s).sum(1) for s in range(vocab_size)], axis=1)
    return _disentanglement(counts, meanings)


def structure_metrics(meanings, messages, vocab_size, max_topsim_objects=1000) -> dict:
    return {
        "topsim": topsim(meanings, messages, max_n=max_topsim_objects),
        "posdis": posdis(meanings, messages),
        "bosdis": bosdis(meanings, messages, vocab_size),
        "n_unique_messages": int(len(np.unique(messages, axis=0))),
    }
