"""Build REPORT.md, summary.json and figures from a results directory.

python -m eibench.report --results results/full
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

from .stats import ci_diff, ci_mean, jump_ratio, verdict_above

# Reference palette (light mode) - slots 1 & 2 are a validated adjacent pair.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2 = "#2a78d6", "#eb6834"


def load(results_dir):
    runs = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if p.endswith("summary.json"):
            continue
        with open(p) as f:
            r = json.load(f)
        c = r["config"]
        runs[c["experiment"]][c["condition"]].append(r)
    return runs


def fmt_ci(ci, d=3):
    return f"{ci[0]:.{d}f} [{ci[1]:.{d}f}, {ci[2]:.{d}f}]"


def hist_at(r, step, key):
    for h in r["history"]:
        if h["step"] == step:
            return h[key]
    raise KeyError(step)


# ---------------------------------------------------------------- criteria
def crit_c1(runs):
    rs = runs.get("c1", {}).get("pair", [])
    if not rs:
        return None
    te = [r["final"]["test_acc"] for r in rs]
    tr = [r["final"]["train_acc"] for r in rs]
    ratio = float(np.mean([a / b if b > 0 else 0 for a, b in zip(te, tr)]))
    ci = ci_mean(te)
    v = "PASS" if (ci[1] >= 0.25 and ratio >= 0.5) else ("INCONCLUSIVE" if (ci[0] >= 0.25 and ratio >= 0.5) else "FAIL")
    return {"verdict": v, "n_seeds": len(rs), "heldout_acc": ci, "train_acc": ci_mean(tr),
            "heldout_over_train": ratio,
            "chance_exact": 1 / (rs[0]["config"]["n_values"] ** rs[0]["config"]["n_attributes"])}


def crit_c4(runs):
    rs = runs.get("c1", {}).get("pair", [])
    if not rs:
        return None
    out = {"n_seeds": len(rs)}
    for k in ("topsim", "posdis", "bosdis"):
        trained = [r["final"]["structure"][k] for r in rs]
        untrained = [r["untrained"][k] for r in rs]
        diffs = np.array(trained) - np.array(untrained)
        out[k] = {"trained": ci_mean(trained), "untrained": ci_mean(untrained), "diff": ci_mean(diffs)}
    t = verdict_above(out["topsim"]["diff"], 0.05)
    pb = [verdict_above(out[k]["diff"], 0.05) for k in ("posdis", "bosdis")]
    pb_v = "PASS" if "PASS" in pb else ("INCONCLUSIVE" if "INCONCLUSIVE" in pb else "FAIL")
    order = ["FAIL", "INCONCLUSIVE", "PASS"]
    out["verdict"] = order[min(order.index(t), order.index(pb_v))]
    return out


def crit_c3(runs):
    g = runs.get("c3", {})
    if not g.get("mixed") or not g.get("isolated"):
        return None
    mx = [r["final"]["test_acc"] for r in g["mixed"]]
    iso = [r["final"]["test_acc"] for r in g["isolated"]]
    d = ci_diff(mx, iso)
    sec = {}
    for k in ("topsim", "posdis", "bosdis"):
        sec[k] = ci_diff([r["final"]["structure"][k] for r in g["mixed"]],
                         [r["final"]["structure"][k] for r in g["isolated"]])
    cross = {c: ci_mean([r["final"]["test"]["exact"]["cross_play"] for r in g[c]]) for c in ("mixed", "isolated")}
    return {"verdict": verdict_above(d), "n_seeds": (len(mx), len(iso)), "mixed": ci_mean(mx),
            "isolated": ci_mean(iso), "diff": d, "secondary_diff": sec, "cross_play": cross}


def crit_c2(runs):
    g = runs.get("c2", {})
    if not g.get("turnover") or not g.get("no_turnover"):
        return None

    def deltas(rs, key="test_acc"):
        return [r["final"][key] - hist_at(r, r["config"]["turnover_start"], key) for r in rs]

    d_t, d_n = deltas(g["turnover"]), deltas(g["no_turnover"])
    d = ci_diff(d_t, d_n)
    ts = ci_diff([r["final"]["structure"]["topsim"] - hist_at(r, r["config"]["turnover_start"], "topsim") for r in g["turnover"]],
                 [r["final"]["structure"]["topsim"] - hist_at(r, r["config"]["turnover_start"], "topsim") for r in g["no_turnover"]])
    return {"verdict": verdict_above(d), "n_seeds": (len(d_t), len(d_n)),
            "delta_turnover": ci_mean(d_t), "delta_none": ci_mean(d_n), "diff": d,
            "topsim_delta_diff": ts,
            "final_turnover": ci_mean([r["final"]["test_acc"] for r in g["turnover"]]),
            "final_none": ci_mean([r["final"]["test_acc"] for r in g["no_turnover"]])}


def crit_c5(runs):
    g = runs.get("c5", {})
    if not g:
        return None
    rows = []
    for cond, rs in g.items():
        h = rs[0]["config"]["hidden"]
        rows.append({"hidden": h, "n": len(rs),
                     "exact": float(np.mean([r["final"]["test_acc"] for r in rs])),
                     "per_attr": float(np.mean([r["final"]["test_per_attr"] for r in rs])),
                     "nll": float(np.mean([r["final"]["test_nll"] for r in rs]))})
    rows.sort(key=lambda x: x["hidden"])
    ex = [r["exact"] for r in rows]
    J_ex = jump_ratio(ex)
    J_pa = jump_ratio([r["per_attr"] for r in rows])
    J_nll = jump_ratio([-r["nll"] for r in rows])
    flat = (max(ex) - min(ex)) < 0.05
    if flat or J_ex < 0.5:
        v = "NO-JUMP"
    elif J_pa >= 0.5:
        v = "PASS"
    elif J_pa < 0.3:
        v = "MIRAGE"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "rows": rows, "J_exact": J_ex, "J_per_attr": J_pa, "J_neg_nll": J_nll}


# ---------------------------------------------------------------- figures
def _style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, fontsize=11, loc="left")
    ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK2, fontsize=9)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)


def figures(runs, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    made = []

    def fig():
        f, ax = plt.subplots(figsize=(7, 3.6), dpi=140)
        f.patch.set_facecolor(SURFACE)
        return f, ax

    # C2 curves
    g = runs.get("c2", {})
    if g.get("turnover") and g.get("no_turnover"):
        f, ax = fig()
        for cond, col, lab in (("turnover", S1, "With turnover"), ("no_turnover", S2, "No turnover")):
            steps = [h["step"] for h in g[cond][0]["history"]]
            m = np.array([[h["test_acc"] for h in r["history"]] for r in g[cond]])
            ax.plot(steps, m.mean(0), color=col, lw=2, label=lab)
            ax.fill_between(steps, m.min(0), m.max(0), color=col, alpha=0.12, lw=0)
        c = g["turnover"][0]["config"]
        ax.axvspan(c["turnover_start"], c["turnover_stop_frac"] * c["steps"], color=GRID, alpha=0.5, lw=0)
        ax.text(c["turnover_start"], 0.97, " turnover window", color=INK2, fontsize=8, va="top")
        ax.set_ylim(0, 1)
        _style(ax, "C2 · Held-out accuracy over training (mean, band = min–max over seeds)", "Step", "Held-out exact accuracy")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        f.tight_layout(); p = os.path.join(out_dir, "c2_curves.png"); f.savefig(p, facecolor=SURFACE); plt.close(f); made.append(p)

    # C3 bars
    g = runs.get("c3", {})
    if g.get("mixed") and g.get("isolated"):
        f, ax = fig()
        labels = ["Mixed population\n(all N² pairs)", "Isolated pairs\n(own pair only)"]
        vals = [[r["final"]["test_acc"] for r in g[c]] for c in ("mixed", "isolated")]
        for i, (v, col) in enumerate(zip(vals, (S1, S2))):
            ax.bar(i, np.mean(v), width=0.5, color=col, edgecolor=SURFACE, linewidth=2)
            ax.scatter(np.full(len(v), i) + np.linspace(-0.08, 0.08, len(v)), v, s=22, color=INK, zorder=3,
                       edgecolor=SURFACE, linewidth=1)
            ax.text(i, np.mean(v) + 0.02, f"{np.mean(v):.3f}", ha="center", color=INK, fontsize=9)
        ax.set_xticks([0, 1], labels)
        ax.set_ylim(0, 1)
        _style(ax, "C3 · Held-out accuracy, same compute (dots = seeds)", "", "Held-out exact accuracy")
        f.tight_layout(); p = os.path.join(out_dir, "c3_bars.png"); f.savefig(p, facecolor=SURFACE); plt.close(f); made.append(p)

    # C5 sweep (accuracies share one 0-1 axis; NLL gets its own chart)
    c5 = crit_c5(runs)
    if c5:
        hs = [r["hidden"] for r in c5["rows"]]
        f, ax = fig()
        ax.plot(hs, [r["exact"] for r in c5["rows"]], color=S1, lw=2, marker="o", ms=6, label="Exact match (all attributes)")
        ax.plot(hs, [r["per_attr"] for r in c5["rows"]], color=S2, lw=2, marker="o", ms=6, label="Per-attribute accuracy (continuous)")
        ax.set_xscale("log", base=2); ax.set_xticks(hs, [str(h) for h in hs]); ax.set_ylim(0, 1)
        _style(ax, f"C5 · Scale sweep — J(exact)={c5['J_exact']:.2f}, J(per-attr)={c5['J_per_attr']:.2f}",
               "Hidden size", "Held-out accuracy")
        ax.legend(frameon=False, fontsize=8, labelcolor=INK)
        f.tight_layout(); p = os.path.join(out_dir, "c5_sweep.png"); f.savefig(p, facecolor=SURFACE); plt.close(f); made.append(p)

        f, ax = fig()
        ax.plot(hs, [r["nll"] for r in c5["rows"]], color=S1, lw=2, marker="o", ms=6)
        ax.set_xscale("log", base=2); ax.set_xticks(hs, [str(h) for h in hs])
        _style(ax, "C5 · Held-out negative log-likelihood per attribute (lower is better)", "Hidden size", "NLL (nats)")
        f.tight_layout(); p = os.path.join(out_dir, "c5_nll.png"); f.savefig(p, facecolor=SURFACE); plt.close(f); made.append(p)
    return made


# ---------------------------------------------------------------- report
def build(results_dir):
    runs = load(results_dir)
    S = {"C1": crit_c1(runs), "C2": crit_c2(runs), "C3": crit_c3(runs), "C4": crit_c4(runs), "C5": crit_c5(runs)}
    figs = figures(runs, results_dir)
    any_run = next((r for e in runs.values() for rs in e.values() for r in rs), None)

    L = ["# EI-Bench v0 — results", ""]
    if any_run:
        n = sum(len(rs) for e in runs.values() for rs in e.values())
        wall = sum(r["wall_s"] for e in runs.values() for rs in e.values() for r in rs)
        L += [f"Runs: **{n}** · total train time {wall / 60:.1f} min · device: `{any_run['device_name']}` · torch {any_run['torch']}", ""]
    L += ["## EI profile", "", "| Criterion | Verdict | Key number |", "|---|---|---|"]
    names = {"C1": "Beyond the objective", "C2": "Ongoing emergence", "C3": "Interaction as source",
             "C4": "Self-organized structure", "C5": "Jumps survive smooth metrics"}
    for k in ("C1", "C2", "C3", "C4", "C5"):
        s = S[k]
        if s is None:
            L.append(f"| {k} {names[k]} | NOT RUN | |"); continue
        key = {"C1": lambda s: f"held-out acc {fmt_ci(s['heldout_acc'])}, ratio {s['heldout_over_train']:.2f}",
               "C2": lambda s: f"Δ(turnover) − Δ(none) = {fmt_ci(s['diff'])}",
               "C3": lambda s: f"mixed − isolated = {fmt_ci(s['diff'])}",
               "C4": lambda s: f"topsim gain {fmt_ci(s['topsim']['diff'])}",
               "C5": lambda s: f"J(exact) {s['J_exact']:.2f}, J(per-attr) {s['J_per_attr']:.2f}"}[k](s)
        L.append(f"| {k} {names[k]} | **{s['verdict']}** | {key} |")
    L += ["", "Values: mean [95% bootstrap CI over seeds]. Thresholds are pre-registered in DEFINITION.md.", ""]

    if S["C1"]:
        s = S["C1"]
        L += ["## C1 · Beyond the objective", "",
              f"- Train accuracy: {fmt_ci(s['train_acc'])}",
              f"- Held-out accuracy (combinations never trained on): {fmt_ci(s['heldout_acc'])}  (chance = {s['chance_exact']:.2e})",
              f"- Held-out / train ratio: {s['heldout_over_train']:.2f}", ""]
    if S["C4"]:
        s = S["C4"]
        L += ["## C4 · Self-organized structure", "", "| Metric | Untrained | Trained | Gain |", "|---|---|---|---|"]
        for k in ("topsim", "posdis", "bosdis"):
            L.append(f"| {k} | {fmt_ci(s[k]['untrained'])} | {fmt_ci(s[k]['trained'])} | {fmt_ci(s[k]['diff'])} |")
        L.append("")
    if S["C3"]:
        s = S["C3"]
        L += ["## C3 · Interaction as source", "", "![C3](c3_bars.png)", "",
              f"- Mixed (all N² pairs): {fmt_ci(s['mixed'])}", f"- Isolated (own pair): {fmt_ci(s['isolated'])}",
              f"- Cross-play accuracy: mixed {fmt_ci(s['cross_play']['mixed'])}, isolated {fmt_ci(s['cross_play']['isolated'])}",
              "- Structure differences (mixed − isolated): " + ", ".join(f"{k} {fmt_ci(v)}" for k, v in s["secondary_diff"].items()), ""]
    if S["C2"]:
        s = S["C2"]
        L += ["## C2 · Ongoing emergence", "", "![C2](c2_curves.png)", "",
              f"- Gain after turnover starts, with turnover: {fmt_ci(s['delta_turnover'])}",
              f"- Gain over the same steps, no turnover: {fmt_ci(s['delta_none'])}",
              f"- Final held-out accuracy: turnover {fmt_ci(s['final_turnover'])}, none {fmt_ci(s['final_none'])}",
              f"- Topsim gain difference: {fmt_ci(s['topsim_delta_diff'])}", ""]
    if S["C5"]:
        s = S["C5"]
        L += ["## C5 · Jumps under smooth metrics", "", "![C5](c5_sweep.png)", "", "![C5 NLL](c5_nll.png)", "",
              "| Hidden | Seeds | Exact | Per-attribute | NLL |", "|---|---|---|---|---|"]
        for r in s["rows"]:
            L.append(f"| {r['hidden']} | {r['n']} | {r['exact']:.3f} | {r['per_attr']:.3f} | {r['nll']:.3f} |")
        L += ["", f"J(exact) = {s['J_exact']:.2f}, J(per-attribute) = {s['J_per_attr']:.2f}, J(−NLL) = {s['J_neg_nll']:.2f}. "
              "J depends on the sweep grid (here: doubling hidden size).", ""]

    with open(os.path.join(results_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    return S, figs


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/full")
    a = p.parse_args(argv)
    S, figs = build(a.results)
    for k, s in S.items():
        print(f"{k}: {s['verdict'] if s else 'NOT RUN'}")
    print(f"wrote {a.results}/REPORT.md, summary.json and {len(figs)} figures")


if __name__ == "__main__":
    main()
