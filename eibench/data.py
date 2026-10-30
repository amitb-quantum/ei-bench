"""Object space and the held-out split."""
from __future__ import annotations

import itertools
import numpy as np
import torch


def all_objects(n_attributes: int, n_values: int) -> np.ndarray:
    """All V^A objects as an (n, A) int array."""
    return np.array(list(itertools.product(range(n_values), repeat=n_attributes)), dtype=np.int64)


def split_objects(objs: np.ndarray, n_values: int, holdout_frac: float, seed: int):
    """Hold out a random subset of *combinations* while guaranteeing every
    (attribute, value) pair still appears in training."""
    rng = np.random.default_rng(seed)
    n = len(objs)
    for _ in range(1000):
        idx = rng.permutation(n)
        n_test = int(round(holdout_frac * n))
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        train = objs[train_idx]
        ok = all(len(np.unique(train[:, a])) == n_values for a in range(objs.shape[1]))
        if ok:
            return train, objs[test_idx]
    raise RuntimeError("could not build a split that covers every value in training")


def one_hot_objects(objs: torch.Tensor, n_values: int) -> torch.Tensor:
    """(B, A) int -> (B, A*V) float."""
    B, A = objs.shape
    x = torch.nn.functional.one_hot(objs, n_values).float()
    return x.reshape(B, A * n_values)
