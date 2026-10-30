"""Population agents with stacked parameters.

Every parameter carries a leading population dimension N, so an entire population
of senders (or receivers) runs as a handful of batched matmuls (torch.baddbmm).
This keeps the GPU busy even though each agent is small, and lets us re-initialise
a single agent (turnover) by resetting one slice.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PopLinear(nn.Module):
    """N independent linear layers: x (N, B, in) -> (N, B, out)."""

    def __init__(self, n: int, d_in: int, d_out: int, bias: bool = True):
        super().__init__()
        self.n, self.d_in, self.d_out = n, d_in, d_out
        self.weight = nn.Parameter(torch.empty(n, d_in, d_out))
        self.bias = nn.Parameter(torch.empty(n, 1, d_out)) if bias else None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, idx: int | None = None):
        bound = 1.0 / math.sqrt(self.d_in)  # matches nn.Linear's default scale
        sl = slice(None) if idx is None else slice(idx, idx + 1)
        self.weight[sl].uniform_(-bound, bound)
        if self.bias is not None:
            self.bias[sl].uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bias is None:
            return torch.bmm(x, self.weight)
        return torch.baddbmm(self.bias, x, self.weight)


class PopGRUCell(nn.Module):
    def __init__(self, n: int, d_in: int, hidden: int):
        super().__init__()
        self.hidden = hidden
        self.wx = PopLinear(n, d_in, 3 * hidden)
        self.wh = PopLinear(n, hidden, 3 * hidden)

    def forward(self, x, h):
        gx = self.wx(x)
        gh = self.wh(h)
        xr, xz, xn = gx.chunk(3, dim=-1)
        hr, hz, hn = gh.chunk(3, dim=-1)
        r = torch.sigmoid(xr + hr)
        z = torch.sigmoid(xz + hz)
        n = torch.tanh(xn + r * hn)
        return (1 - z) * n + z * h


def _reset_module(module: nn.Module, idx: int | None):
    for m in module.modules():
        if isinstance(m, PopLinear):
            m.reset_parameters(idx)


def gumbel_st(logits: torch.Tensor, tau: float) -> torch.Tensor:
    """Straight-through Gumbel-softmax: one-hot forward, soft gradient backward."""
    return F.gumbel_softmax(logits, tau=tau, hard=True, dim=-1)


class SenderPop(nn.Module):
    def __init__(self, n, d_obj, hidden, embed, vocab, msg_len):
        super().__init__()
        self.n, self.vocab, self.msg_len = n, vocab, msg_len
        self.enc = PopLinear(n, d_obj, hidden)
        self.sos = nn.Parameter(torch.zeros(n, 1, embed))
        self.cell = PopGRUCell(n, embed, hidden)
        self.out = PopLinear(n, hidden, vocab)
        self.emb = PopLinear(n, vocab, embed, bias=False)

    def forward(self, x: torch.Tensor, tau: float = 1.0, greedy: bool = False):
        """x: (B, d_obj) shared by all senders -> messages (N, B, L, K) one-hot."""
        B = x.shape[0]
        xs = x.unsqueeze(0).expand(self.n, B, x.shape[1])
        h = torch.tanh(self.enc(xs))
        inp = self.sos.expand(self.n, B, self.sos.shape[-1])
        symbols = []
        for _ in range(self.msg_len):
            h = self.cell(inp, h)
            logits = self.out(h)
            if greedy:
                s = F.one_hot(logits.argmax(-1), self.vocab).float()
            else:
                s = gumbel_st(logits, tau)
            symbols.append(s)
            inp = self.emb(s)
        return torch.stack(symbols, dim=2)

    @torch.no_grad()
    def reset_agent(self, idx: int):
        _reset_module(self, idx)
        self.sos[idx].zero_()


class ReceiverPop(nn.Module):
    def __init__(self, n, n_attributes, n_values, hidden, embed, vocab):
        super().__init__()
        self.n, self.A, self.V = n, n_attributes, n_values
        self.emb = PopLinear(n, vocab, embed, bias=False)
        self.cell = PopGRUCell(n, embed, hidden)
        self.out = PopLinear(n, hidden, n_attributes * n_values)
        self.hidden = hidden

    def forward(self, msgs: torch.Tensor) -> torch.Tensor:
        """msgs: (N, B, L, K) -> logits (N, B, A, V); receiver j reads msgs[j]."""
        N, B, L, K = msgs.shape
        h = msgs.new_zeros(N, B, self.hidden)
        for t in range(L):
            h = self.cell(self.emb(msgs[:, :, t]), h)
        return self.out(h).reshape(N, B, self.A, self.V)

    @torch.no_grad()
    def reset_agent(self, idx: int):
        _reset_module(self, idx)


@torch.no_grad()
def perturb_agent(module: nn.Module, idx: int, generator: torch.Generator | None = None) -> dict:
    """Displacement-matched control for turnover: for every parameter tensor, add isotropic
    Gaussian noise to agent `idx` whose norm equals ||fresh_init - current||, i.e. the distance
    a reset would have moved it. Returns per-tensor displacement norms (for logging)."""
    params = list(module.parameters())
    old = [p[idx].clone() for p in params]
    module.reset_agent(idx)                      # writes a fresh sample into slice idx
    norms = {}
    for k, (p, o) in enumerate(zip(params, old)):
        d = (p[idx] - o).norm()
        noise = torch.randn(o.shape, device=o.device, dtype=o.dtype, generator=generator)
        nn_ = noise.norm()
        p[idx].copy_(o + noise * (d / nn_) if nn_ > 0 else o)
        norms[k] = float(d)
    return norms


@torch.no_grad()
def shrink_perturb_agent(module: nn.Module, idx: int, sigma: float = 0.1) -> dict:
    """Plasticity-restoring control (shrink-and-perturb): for every parameter tensor of agent
    `idx`, w' = lam * w + sigma * f, where f is a fresh-init sample and lam = ||f|| / ||w||, so the
    shrunk part has fresh-init norm and keeps the learned direction. Zero-init tensors (fresh value
    0) become 0, as in a reset. Returns per-tensor diagnostics (norms, cosine to old)."""
    params = list(module.parameters())
    old = [p[idx].clone() for p in params]
    module.reset_agent(idx)                      # writes a fresh sample f into slice idx
    info = {}
    for k, (p, o) in enumerate(zip(params, old)):
        f = p[idx].clone()
        nf, no = f.norm(), o.norm()
        lam = (nf / no) if no > 0 else torch.zeros((), device=o.device)
        new = lam * o + sigma * f
        p[idx].copy_(new)
        cos = float((new.flatten() @ o.flatten()) / (new.norm() * no)) if (new.norm() > 0 and no > 0) else 0.0
        info[k] = {"old_norm": float(no), "fresh_norm": float(nf), "new_norm": float(new.norm()),
                   "lam": float(lam), "cos_to_old": cos}
    return info


def reset_optimizer_slice(opt: torch.optim.Optimizer, module: nn.Module, idx: int):
    """Zero Adam moments for one agent's slice so a re-initialised agent truly starts fresh."""
    for p in module.parameters():
        st = opt.state.get(p)
        if not st:
            continue
        for key in ("exp_avg", "exp_avg_sq"):
            if key in st:
                st[key][idx].zero_()
