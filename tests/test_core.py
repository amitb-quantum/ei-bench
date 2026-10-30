import numpy as np
import torch

from eibench.agents import ReceiverPop, SenderPop, reset_optimizer_slice
from eibench.data import all_objects, split_objects
from eibench.metrics import bosdis, posdis, topsim
from eibench.stats import jump_ratio


def test_split_covers_all_values():
    objs = all_objects(4, 8)
    tr, te = split_objects(objs, 8, 0.1, 0)
    assert len(tr) + len(te) == 8 ** 4
    assert not set(map(tuple, tr)) & set(map(tuple, te))
    for a in range(4):
        assert len(np.unique(tr[:, a])) == 8


def test_perfect_compositional_language_scores_high():
    m = all_objects(3, 5)
    msgs = m.copy()  # symbol j encodes attribute j
    assert topsim(m, msgs) > 0.99
    assert posdis(m, msgs) > 0.99
    assert bosdis(m, msgs + np.arange(3) * 5, vocab_size=15) > 0.99


def test_random_language_scores_low():
    rng = np.random.default_rng(0)
    m = all_objects(3, 5)
    msgs = rng.integers(0, 10, size=(len(m), 4))
    assert abs(topsim(m, msgs)) < 0.1
    assert posdis(m, msgs) < 0.15


def test_population_shapes_and_reset():
    S = SenderPop(3, 32, 16, 8, 10, 5)
    R = ReceiverPop(3, 4, 8, 16, 8, 10)
    x = torch.rand(7, 32)
    m = S(x)
    assert m.shape == (3, 7, 5, 10) and torch.allclose(m.sum(-1), torch.ones(3, 7, 5))
    assert R(m).shape == (3, 7, 4, 8)
    opt = torch.optim.Adam(list(S.parameters()) + list(R.parameters()))
    R(S(x)).sum().backward(); opt.step()
    before = S.enc.weight[0].clone(), S.enc.weight[1].clone()
    S.reset_agent(1); reset_optimizer_slice(opt, S, 1)
    assert torch.equal(S.enc.weight[0], before[0])
    assert not torch.equal(S.enc.weight[1], before[1])
    assert opt.state[S.enc.weight]["exp_avg"][1].abs().sum() == 0


def test_jump_ratio():
    assert jump_ratio([0, 0, 0, 1]) == 1.0
    assert abs(jump_ratio([0, 0.25, 0.5, 0.75, 1]) - 0.25) < 1e-9
