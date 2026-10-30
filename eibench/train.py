"""Training and evaluation for one run."""
from __future__ import annotations

import json
import math
import os
import platform
import time

import numpy as np
import torch
import torch.nn.functional as F

from .agents import ReceiverPop, SenderPop, perturb_agent, reset_optimizer_slice, shrink_perturb_agent
from .config import Config
from .data import all_objects, one_hot_objects, split_objects
from .metrics import structure_metrics


def pick_device(pref: str = "auto") -> torch.device:
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(pref)


@torch.no_grad()
def evaluate(sender, receiver, objs: torch.Tensor, cfg: Config) -> dict:
    """Evaluate every sender-receiver pairing on `objs` with greedy (noise-free) messages."""
    N, V = cfg.pop_size, cfg.n_values
    x = one_hot_objects(objs, V)
    msgs = sender(x, greedy=True)  # (N, n, L, K)
    exact = np.zeros((N, N))  # [receiver, sender]
    per_attr = np.zeros((N, N))
    nll = np.zeros((N, N))
    ar = torch.arange(N, device=objs.device)
    for s in range(N):
        perm = (ar + s) % N  # receiver j reads sender perm[j]
        logits = receiver(msgs[perm])  # (N, n, A, V)
        pred = logits.argmax(-1)
        tgt = objs.unsqueeze(0).expand_as(pred)
        correct = (pred == tgt)
        lp = F.log_softmax(logits, -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # (N, n, A)
        for j in range(N):
            k = int(perm[j])
            exact[j, k] = correct[j].all(-1).float().mean().item()
            per_attr[j, k] = correct[j].float().mean().item()
            nll[j, k] = (-lp[j]).mean().item()
    diag = np.eye(N, dtype=bool)

    def summ(m):
        out = {"all_pairs": float(m.mean()), "own_pair": float(m[diag].mean())}
        out["cross_play"] = float(m[~diag].mean()) if N > 1 else float("nan")
        return out

    return {"exact": summ(exact), "per_attr": summ(per_attr), "nll": summ(nll),
            "exact_matrix": exact.tolist()}


@torch.no_grad()
def language_structure(sender, objs_np: np.ndarray, cfg: Config, device) -> dict:
    objs = torch.as_tensor(objs_np, device=device)
    msgs = sender(one_hot_objects(objs, cfg.n_values), greedy=True).argmax(-1).cpu().numpy()  # (N, n, L)
    per = [structure_metrics(objs_np, msgs[i], cfg.vocab_size, cfg.max_topsim_objects)
           for i in range(cfg.pop_size)]
    keys = per[0].keys()
    return {k: float(np.mean([p[k] for p in per])) for k in keys} | {"per_sender": per}


def capability(ev: dict, cfg: Config, metric: str = "exact") -> float:
    """Pre-registered primary score: isolated pairs are scored on their own pair only,
    mixed populations on all N^2 pairs (a conservative comparison for 'mixed')."""
    return ev[metric]["own_pair" if cfg.mixing == "isolated" else "all_pairs"]


def train_run(cfg: Config, device: str = "auto", verbose: bool = True) -> dict:
    dev = pick_device(device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed + 12345)

    objs = all_objects(cfg.n_attributes, cfg.n_values)
    train_np, test_np = split_objects(objs, cfg.n_values, cfg.holdout_frac, cfg.split_seed)
    train_t = torch.as_tensor(train_np, device=dev)
    test_t = torch.as_tensor(test_np, device=dev)
    d_obj = cfg.n_attributes * cfg.n_values

    sender = SenderPop(cfg.pop_size, d_obj, cfg.hidden, cfg.embed, cfg.vocab_size, cfg.msg_len).to(dev)
    receiver = ReceiverPop(cfg.pop_size, cfg.n_attributes, cfg.n_values, cfg.hidden, cfg.embed,
                           cfg.vocab_size).to(dev)
    N = cfg.pop_size
    opt = torch.optim.Adam(list(sender.parameters()) + list(receiver.parameters()), lr=cfg.lr)
    decay_start = int((1 - cfg.lr_decay_frac) * cfg.steps)

    def lr_mult(step):  # constant, then linear decay so the final snapshot is stable
        if cfg.lr_decay_frac <= 0 or step < decay_start:
            return 1.0
        frac = (step - decay_start) / max(1, cfg.steps - decay_start)
        return 1.0 - frac * (1.0 - cfg.lr_final_mult)

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_mult)

    # per-agent learning-rate multipliers (heterogeneous populations). Adam is invariant to
    # gradient scale, so we scale each agent's *update* instead: w <- w_old + m_i * (w_adam - w_old)
    hetero = cfg.lr_spread != 1.0 and N > 1
    if hetero:
        k = cfg.lr_spread
        m = torch.tensor(np.geomspace(1.0 / k, k, N), dtype=torch.float32, device=dev)
        mult_s, mult_r = m.view(N, 1, 1), m.flip(0).view(N, 1, 1)
        het_params = [(p, mult_s) for p in sender.parameters()] + [(p, mult_r) for p in receiver.parameters()]

    ident = torch.arange(N, device=dev)
    turnover_stop = int(cfg.turnover_stop_frac * cfg.steps)
    eval_steps = set(range(0, cfg.steps + 1, cfg.eval_every)) | {cfg.steps}
    if 0 < cfg.turnover_start <= cfg.steps:
        eval_steps.add(cfg.turnover_start)  # C2 baseline point (also logged for the control)

    history, events = [], []
    untrained = None
    t0 = time.time()

    def do_eval(step):
        sender.eval(); receiver.eval()
        ev_tr = evaluate(sender, receiver, train_t, cfg)
        ev_te = evaluate(sender, receiver, test_t, cfg)
        st = language_structure(sender, objs, cfg, dev)
        sender.train(); receiver.train()
        rec = {
            "step": step,
            "train_acc": capability(ev_tr, cfg),
            "test_acc": capability(ev_te, cfg),
            "test_per_attr": capability(ev_te, cfg, "per_attr"),
            "test_nll": capability(ev_te, cfg, "nll"),
            "test_cross_play": ev_te["exact"]["cross_play"],
            "test_own_pair": ev_te["exact"]["own_pair"],
            "test_all_pairs": ev_te["exact"]["all_pairs"],
            "topsim": st["topsim"], "posdis": st["posdis"], "bosdis": st["bosdis"],
            "n_unique_messages": st["n_unique_messages"],
            "elapsed_s": time.time() - t0,
        }
        return rec, ev_te, st

    for step in range(cfg.steps + 1):
        if step in eval_steps:
            rec, ev_te, st = do_eval(step)
            history.append(rec)
            if step == 0:
                untrained = rec | {"structure_per_sender": st["per_sender"]}
            if verbose:
                print(f"[{cfg.run_id()}] step {step:6d} train {rec['train_acc']:.3f} "
                      f"test {rec['test_acc']:.3f} attr {rec['test_per_attr']:.3f} "
                      f"topsim {rec['topsim']:.3f} ({rec['elapsed_s']:.0f}s)", flush=True)
        if step == cfg.steps:
            break

        # turnover: re-initialise one random agent (sender or receiver)
        if (cfg.turnover_every and step > cfg.turnover_start and step < turnover_stop
                and (step - cfg.turnover_start) % cfg.turnover_every == 0):
            i = int(rng.integers(N))
            role = "sender" if rng.random() < 0.5 else "receiver"
            mod = sender if role == "sender" else receiver
            sp = None
            if cfg.turnover_mode == "perturb":
                disp = perturb_agent(mod, i)
            elif cfg.turnover_mode == "shrink_perturb":
                sp = shrink_perturb_agent(mod, i, sigma=cfg.sp_sigma); disp = None
            else:
                mod.reset_agent(i); disp = None
            reset_optimizer_slice(opt, mod, i)
            ev = {"step": step, "agent": i, "role": role, "mode": cfg.turnover_mode}
            if disp is not None:
                ev["displacement_norm"] = float(sum(v * v for v in disp.values()) ** 0.5)
            if sp is not None:
                ev["sp_mean_cos_to_old"] = float(np.mean([v["cos_to_old"] for v in sp.values() if v["old_norm"] > 0]))
                ev["sp_norm_ratio_new_over_old"] = float(
                    sum(v["new_norm"] ** 2 for v in sp.values()) ** 0.5 / max(1e-12, sum(v["old_norm"] ** 2 for v in sp.values()) ** 0.5))
            events.append(ev)

        idx = torch.randint(len(train_t), (cfg.batch_size,), device=dev)
        tgt = train_t[idx]
        msgs = sender(one_hot_objects(tgt, cfg.n_values), tau=cfg.gumbel_tau)
        perm = torch.randperm(N, device=dev) if cfg.mixing == "mixed" else ident
        logits = receiver(msgs[perm])
        loss = F.cross_entropy(logits.reshape(-1, cfg.n_values),
                               tgt.unsqueeze(0).expand(N, -1, -1).reshape(-1))
        loss = loss * N * cfg.n_attributes  # sum over agents & attributes, mean over batch
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if hetero:
            olds = [p.detach().clone() for p, _ in het_params]
        opt.step()
        if hetero:
            with torch.no_grad():
                for (p, mlt), o in zip(het_params, olds):
                    p.copy_(o + mlt * (p - o))
        sched.step()

    final_tr = evaluate(sender, receiver, train_t, cfg)
    final_te = evaluate(sender, receiver, test_t, cfg)
    final_st = language_structure(sender, objs, cfg, dev)
    return {
        "run_id": cfg.run_id(),
        "config": cfg.to_dict(),
        "device": str(dev),
        "device_name": torch.cuda.get_device_name(dev) if dev.type == "cuda" else platform.processor(),
        "torch": torch.__version__,
        "wall_s": time.time() - t0,
        "untrained": untrained,
        "history": history,
        "turnover_events": events,
        "final": {
            "train": final_tr, "test": final_te,
            "structure": {k: v for k, v in final_st.items() if k != "per_sender"},
            "structure_per_sender": final_st["per_sender"],
            "train_acc": capability(final_tr, cfg),
            "test_acc": capability(final_te, cfg),
            "test_per_attr": capability(final_te, cfg, "per_attr"),
            "test_nll": capability(final_te, cfg, "nll"),
        },
    }


def save_result(res: dict, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, res["run_id"] + ".json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(res, f, indent=1)
    os.replace(tmp, path)
    return path
