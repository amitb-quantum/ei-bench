"""Run configuration for EI-Bench."""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
import hashlib
import json


@dataclass
class Config:
    # task
    n_attributes: int = 4
    n_values: int = 8
    holdout_frac: float = 0.10
    split_seed: int = 0  # same held-out set for every run

    # channel
    msg_len: int = 6
    vocab_size: int = 16
    gumbel_tau: float = 1.0

    # agents
    hidden: int = 128
    embed: int = 32
    pop_size: int = 1
    mixing: str = "mixed"  # "mixed" (shuffled pairings) or "isolated" (fixed pairs)

    # optimisation
    steps: int = 8000
    batch_size: int = 256
    lr: float = 1e-3
    lr_decay_frac: float = 0.2   # linear decay to lr_final_mult*lr over the last fraction of steps
    lr_final_mult: float = 0.05
    seed: int = 0

    # turnover (iterated learning); 0 disables
    turnover_every: int = 0
    turnover_start: int = 0       # step at which turnover begins
    turnover_stop_frac: float = 0.8  # stop resetting after this fraction of steps (settle phase)
    turnover_mode: str = "reset"     # "reset" (new agent) or "perturb" (displacement-matched noise)

    # heterogeneity: per-agent learning-rate multipliers geometrically spaced in [1/k, k]; 1 = homogeneous
    lr_spread: float = 1.0

    # logging
    eval_every: int = 500
    max_topsim_objects: int = 1000

    # bookkeeping
    experiment: str = "adhoc"
    condition: str = ""
    tags: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def run_id(self) -> str:
        d = self.to_dict()
        h = hashlib.sha1(json.dumps(d, sort_keys=True).encode()).hexdigest()[:10]
        cond = f"_{self.condition}" if self.condition else ""
        return f"{self.experiment}{cond}_s{self.seed}_{h}"
