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


def test_perturb_matches_reset_displacement():
    from eibench.agents import perturb_agent
    torch.manual_seed(0)
    S = SenderPop(3, 32, 16, 8, 10, 5)
    with torch.no_grad():
        for p in S.parameters():
            p.add_(0.3)  # move away from init so a reset would displace it
    before = [p.detach().clone() for p in S.parameters()]
    norms = perturb_agent(S, 1)
    for k, (p, b) in enumerate(zip(S.parameters(), before)):
        assert torch.equal(p[0], b[0]) and torch.equal(p[2], b[2])   # other agents untouched
        moved = (p[1] - b[1]).norm().item()
        assert abs(moved - norms[k]) < 1e-4 * max(1.0, norms[k])     # same distance as a reset


def test_transition_width():
    from eibench.report import transition_width
    x = np.arange(8.0)
    step = [0, 0, 0, 0, 1, 1, 1, 1]
    ramp = np.linspace(0, 1, 8)
    assert transition_width(x, step) < 1.0
    assert abs(transition_width(x, ramp) - 0.8 * 7) < 1e-6


def test_v0_width_reference():
    """The worked example quoted in DEFINITION.md section 7."""
    from eibench.report import transition_width
    h = np.log2([4, 8, 16, 32, 64, 128, 256])
    ex = [0.002, 0.001, 0.007, 0.022, 0.410, 0.988, 0.996]
    pa = [0.265, 0.303, 0.446, 0.607, 0.836, 0.997, 0.999]
    we, wc = transition_width(h, ex), transition_width(h, pa)
    assert 1.4 < we < 1.8 and 3.0 < wc < 3.5


def test_heterogeneous_lr_scales_updates():
    """Agent i's parameter change per step should scale with its multiplier (Adam is scale-free)."""
    from eibench.config import Config
    from eibench import train as T
    cfg = Config(steps=1, eval_every=1000, pop_size=4, lr_spread=2.0, lr_decay_frac=0.0, hidden=16, embed=8)
    orig, orig_ls = T.evaluate, T.language_structure
    import eibench.agents as A
    torch.manual_seed(0)
    # run one step manually through train_run's machinery by monkeypatching evaluate out
    T.evaluate = lambda *a, **k: {m: {"all_pairs": 0.0, "own_pair": 0.0, "cross_play": 0.0} for m in ("exact", "per_attr", "nll")} | {"exact_matrix": []}
    T.language_structure = lambda *a, **k: {"topsim": 0, "posdis": 0, "bosdis": 0, "n_unique_messages": 0, "per_sender": []}
    real_init = A.SenderPop.__init__
    created = []
    def spy(self, *a, **k):
        real_init(self, *a, **k); created.append(self)
    A.SenderPop.__init__ = spy
    T.SenderPop = A.SenderPop
    try:
        torch.manual_seed(0)
        T.train_run(cfg, device="cpu", verbose=False)
    finally:
        A.SenderPop.__init__ = real_init
        T.evaluate, T.language_structure = orig, orig_ls
    S = created[0]
    # first Adam step moves each weight by ~lr (sign of grad); so mean |delta| ~ lr * multiplier
    torch.manual_seed(0)
    fresh = A.SenderPop(4, cfg.n_attributes * cfg.n_values, 16, 8, cfg.vocab_size, cfg.msg_len)
    d = (S.out.weight - fresh.out.weight).abs().mean(dim=(1, 2))
    ratio = (d[-1] / d[0]).item()
    assert 3.0 < ratio < 5.0, ratio   # multipliers 0.5 .. 2.0 -> ratio 4
