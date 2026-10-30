"""Build REPORT.md, summary.json and figures from a results directory (v0 or v0.2 suite).

python -m eibench.report --results results/v02
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy.stats import mannwhitneyu

from .stats import ci_diff, ci_mean, jump_ratio, verdict_above

# Reference palette (light). Slots 1-3 validate all-pairs; slots 1-2 are an adjacent pair.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
ORDER = ["FAIL", "INCONCLUSIVE", "PASS"]


def load(results_dir):
    runs = defaultdict(lambda: defaultdict(list))
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        if os.path.basename(p) == "summary.json":
            continue
        with open(p) as f:
            r = json.load(f)
        c = r["config"]
        runs[c["experiment"]][c["condition"]].append(r)
    return runs


def fmt_ci(ci, d=3):
    return f"{ci[0]:.{d}f} [{ci[1]:.{d}f}, {ci[2]:.{d}f}]"


def mw(a, b):
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    return float(mannwhitneyu(a, b, alternative="two-sided").pvalue)


def hist_at(r, step, key):
    for h in r["history"]:
        if h["step"] == step:
            return h[key]
    raise KeyError(step)


def final(rs, key="test_acc"):
    return [r["final"][key] for r in rs]


# ---------------------------------------------------------------- criteria
def crit_c1(runs):
    rs = runs.get("c1", {}).get("pair", [])
    if not rs:
        return None
    te, tr = final(rs), final(rs, "train_acc")
    ratio = float(np.mean([a / b if b > 0 else 0 for a, b in zip(te, tr)]))
    ci = ci_mean(te)
    v = "PASS" if (ci[1] >= 0.25 and ratio >= 0.5) else ("INCONCLUSIVE" if (ci[0] >= 0.25 and ratio >= 0.5) else "FAIL")
    c = rs[0]["config"]
    return {"verdict": v, "n_seeds": len(rs), "heldout_acc": ci, "train_acc": ci_mean(tr),
            "heldout_over_train": ratio, "chance_exact": 1 / (c["n_values"] ** c["n_attributes"]),
            "per_seed": te}


def crit_c4(runs):
    rs = runs.get("c1", {}).get("pair", [])
    if not rs:
        return None
    out = {"n_seeds": len(rs)}
    for k in ("topsim", "posdis", "bosdis"):
        trained = [r["final"]["structure"][k] for r in rs]
        untrained = [r["untrained"][k] for r in rs]
        out[k] = {"trained": ci_mean(trained), "untrained": ci_mean(untrained),
                  "diff": ci_mean(np.array(trained) - np.array(untrained))}
    t = verdict_above(out["topsim"]["diff"], 0.05)
    pb = [verdict_above(out[k]["diff"], 0.05) for k in ("posdis", "bosdis")]
    pb_v = "PASS" if "PASS" in pb else ("INCONCLUSIVE" if "INCONCLUSIVE" in pb else "FAIL")
    out["verdict"] = ORDER[min(ORDER.index(t), ORDER.index(pb_v))]
    return out


def _deltas(rs, key="test_acc"):
    return [r["final"][key] - hist_at(r, r["config"]["turnover_start"], key) for r in rs]


def _topsim_deltas(rs):
    return [r["final"]["structure"]["topsim"] - hist_at(r, r["config"]["turnover_start"], "topsim") for r in rs]


def crit_turnover(runs, exp, arm_a, arm_b):
    """Δ held-out accuracy (end minus turnover start), arm_a minus arm_b."""
    g = runs.get(exp, {})
    if not g.get(arm_a) or not g.get(arm_b):
        return None
    da, db = _deltas(g[arm_a]), _deltas(g[arm_b])
    d = ci_diff(da, db)
    return {"verdict": verdict_above(d), "n_seeds": (len(da), len(db)), "arms": (arm_a, arm_b),
            "delta_a": ci_mean(da), "delta_b": ci_mean(db), "diff": d, "mannwhitney_p_delta": mw(da, db),
            "final_a": ci_mean(final(g[arm_a])), "final_b": ci_mean(final(g[arm_b])),
            "mannwhitney_p_final": mw(final(g[arm_a]), final(g[arm_b])),
            "topsim_delta_diff": ci_diff(_topsim_deltas(g[arm_a]), _topsim_deltas(g[arm_b])),
            "per_seed_final": {arm_a: final(g[arm_a]), arm_b: final(g[arm_b])}}


def crit_mix(runs, exp, mixed, isolated):
    g = runs.get(exp, {})
    if not g.get(mixed) or not g.get(isolated):
        return None
    mx, iso = final(g[mixed]), final(g[isolated])
    d = ci_diff(mx, iso)
    sec = {k: ci_diff([r["final"]["structure"][k] for r in g[mixed]],
                      [r["final"]["structure"][k] for r in g[isolated]]) for k in ("topsim", "posdis", "bosdis")}
    cross = {c: ci_mean([r["final"]["test"]["exact"]["cross_play"] for r in g[c]]) for c in (mixed, isolated)}
    return {"verdict": verdict_above(d), "n_seeds": (len(mx), len(iso)), "mixed": ci_mean(mx),
            "isolated": ci_mean(iso), "diff": d, "mannwhitney_p": mw(mx, iso), "secondary_diff": sec,
            "cross_play": cross}


def _curve_rows(g):
    rows = []
    for cond, rs in g.items():
        rows.append({"hidden": rs[0]["config"]["hidden"], "n": len(rs),
                     "exact_seeds": final(rs), "per_attr_seeds": final(rs, "test_per_attr"),
                     "nll_seeds": final(rs, "test_nll")})
    rows.sort(key=lambda x: x["hidden"])
    for r in rows:
        r["exact"] = float(np.mean(r["exact_seeds"]))
        r["per_attr"] = float(np.mean(r["per_attr_seeds"]))
        r["nll"] = float(np.mean(r["nll_seeds"]))
    return rows


def crit_c5(runs):
    g = runs.get("c5", {})
    if not g:
        return None
    rows = _curve_rows(g)
    ex = [r["exact"] for r in rows]
    J_ex, J_pa = jump_ratio(ex), jump_ratio([r["per_attr"] for r in rows])
    J_nll = jump_ratio([-r["nll"] for r in rows])
    if (max(ex) - min(ex)) < 0.05 or J_ex < 0.5:
        v = "NO-JUMP"
    elif J_pa >= 0.5:
        v = "PASS"
    elif J_pa < 0.3:
        v = "MIRAGE"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "rows": rows, "J_exact": J_ex, "J_per_attr": J_pa, "J_neg_nll": J_nll}


def transition_width(x, y, lo=0.1, hi=0.9):
    """10-90% width (in units of x) of the curve y(x) normalized to its own min..max."""
    y = np.asarray(y, float)
    rng = y.max() - y.min()
    if rng < 1e-9:
        return float("nan")
    z = (y - y.min()) / rng

    def cross(level):
        for i in range(len(z)):
            if z[i] >= level:
                if i == 0:
                    return x[0]
                return x[i - 1] + (level - z[i - 1]) / (z[i] - z[i - 1]) * (x[i] - x[i - 1])
        return x[-1]
    return float(cross(hi) - cross(lo))


def crit_c5b(runs, n_boot=2000, seed=0):
    g = runs.get("c5b", {})
    if not g:
        return None
    rows = _curve_rows(g)
    x = np.log2([r["hidden"] for r in rows])
    ex = np.array([r["exact"] for r in rows]); pa = np.array([r["per_attr"] for r in rows])
    we, wc = transition_width(x, ex), transition_width(x, pa)
    R = wc / we if we and we > 0 else float("nan")
    rng = np.random.default_rng(seed)
    Rs, wes, wcs = [], [], []
    for _ in range(n_boot):
        e_b = [np.mean(rng.choice(r["exact_seeds"], len(r["exact_seeds"]))) for r in rows]
        p_b = [np.mean(rng.choice(r["per_attr_seeds"], len(r["per_attr_seeds"]))) for r in rows]
        a, b = transition_width(x, e_b), transition_width(x, p_b)
        wes.append(a); wcs.append(b)
        if a and a > 0 and np.isfinite(b):
            Rs.append(b / a)
    q = lambda v: (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975)))
    R_ci = q(Rs) if Rs else (float("nan"), float("nan"))
    if not np.isfinite(we) or we > 2.5 or (ex.max() - ex.min()) < 0.05:
        v = "NO-JUMP"
    elif R >= 2 and R_ci[0] > 1.5:
        v = "MIRAGE"
    elif R <= 1.5 and R_ci[1] < 2:
        v = "PASS"
    else:
        v = "INCONCLUSIVE"
    return {"verdict": v, "rows": [{k: r[k] for k in ("hidden", "n", "exact", "per_attr", "nll")} for r in rows],
            "w_exact": we, "w_exact_ci": q(wes), "w_per_attr": wc, "w_per_attr_ci": q(wcs),
            "R": R, "R_ci": R_ci, "J_exact": jump_ratio(ex), "J_per_attr": jump_ratio(pa)}


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


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _new(plt, w=7, h=3.6):
    f, ax = plt.subplots(figsize=(w, h), dpi=140)
    f.patch.set_facecolor(SURFACE)
    return f, ax


def _save(plt, f, out_dir, name, made):
    f.tight_layout()
    p = os.path.join(out_dir, name)
    f.savefig(p, facecolor=SURFACE)
    plt.close(f)
    made.append(p)


def fig_turnover(plt, g, arms, title, out_dir, name, made):
    arms = [a for a in arms if g.get(a[0])]
    if len(arms) < 2:
        return
    f, ax = _new(plt)
    for cond, col, lab in arms:
        steps = [h["step"] for h in g[cond][0]["history"]]
        m = np.array([[h["test_acc"] for h in r["history"]] for r in g[cond]])
        ax.plot(steps, m.mean(0), color=col, lw=2, label=lab)
        ax.fill_between(steps, m.min(0), m.max(0), color=col, alpha=0.12, lw=0)
    c = g[arms[0][0]][0]["config"]
    ax.axvspan(c["turnover_start"], c["turnover_stop_frac"] * c["steps"], color=GRID, alpha=0.5, lw=0)
    ax.text(c["turnover_start"], 0.03, " turnover / perturbation window", color=INK2, fontsize=8, va="bottom")
    ax.set_ylim(0, 1.02)
    _style(ax, title, "Step", "Held-out exact accuracy")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    _save(plt, f, out_dir, name, made)


def fig_bars(plt, groups, title, out_dir, name, made):
    """groups: list of (label, [(arm_label, values, color), ...])"""
    f, ax = _new(plt, w=max(5, 2.4 * len(groups) + 1.5))
    x, ticks, labels, legend_done = 0.0, [], [], set()
    for glabel, arms in groups:
        start = x
        for alabel, vals, col in arms:
            ax.bar(x, np.mean(vals), width=0.8, color=col, edgecolor=SURFACE, linewidth=2,
                   label=alabel if alabel not in legend_done else None)
            legend_done.add(alabel)
            ax.scatter(np.full(len(vals), x) + np.linspace(-0.2, 0.2, len(vals)), vals, s=14, color=INK,
                       zorder=3, edgecolor=SURFACE, linewidth=0.8)
            ax.text(x, min(1.0, np.mean(vals)) + 0.02, f"{np.mean(vals):.3f}", ha="center", color=INK, fontsize=8)
            x += 1
        ticks.append((start + x - 1) / 2); labels.append(glabel)
        x += 0.8
    ax.set_xticks(ticks, labels)
    ax.set_ylim(0, 1.08)
    _style(ax, title, "", "Held-out exact accuracy")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=2)
    _save(plt, f, out_dir, name, made)


def fig_sweep(plt, s, exp, out_dir, made):
    hs = [r["hidden"] for r in s["rows"]]
    f, ax = _new(plt)
    ax.plot(hs, [r["exact"] for r in s["rows"]], color=S1, lw=2, marker="o", ms=6, label="Exact match (all attributes)")
    ax.plot(hs, [r["per_attr"] for r in s["rows"]], color=S2, lw=2, marker="o", ms=6, label="Per-attribute accuracy (continuous)")
    ax.set_xscale("log", base=2); ax.set_xticks(hs, [str(h) for h in hs]); ax.set_ylim(0, 1.02)
    if exp == "c5b":
        t = f"C5b · Scale sweep — transition width exact {s['w_exact']:.2f}, per-attr {s['w_per_attr']:.2f} octaves"
    else:
        t = f"C5 · Scale sweep — J(exact)={s['J_exact']:.2f}, J(per-attr)={s['J_per_attr']:.2f}"
    _style(ax, t, "Hidden size", "Held-out accuracy")
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    _save(plt, f, out_dir, f"{exp}_sweep.png", made)

    f, ax = _new(plt)
    ax.plot(hs, [r["nll"] for r in s["rows"]], color=S1, lw=2, marker="o", ms=6)
    ax.set_xscale("log", base=2); ax.set_xticks(hs, [str(h) for h in hs])
    _style(ax, f"{exp.upper()} · Held-out NLL per attribute (lower is better)", "Hidden size", "NLL (nats)")
    _save(plt, f, out_dir, f"{exp}_nll.png", made)


# ---------------------------------------------------------------- report
NAMES = {"C1": "Beyond the objective", "C2": "Ongoing emergence", "C2b": "Ongoing emergence: reset vs. perturbation",
         "C3": "Interaction as source", "C3b": "Interaction: heterogeneous population",
         "C4": "Self-organized structure", "C5": "Jumps survive smooth metrics", "C5b": "Jumps: transition width"}


def build(results_dir):
    runs = load(results_dir)
    v02 = "c2b" in runs or "c3b" in runs or "c5b" in runs
    S = {"C1": crit_c1(runs), "C4": crit_c4(runs)}
    if v02:
        S["C2"] = crit_turnover(runs, "c2b", "turnover", "no_turnover")
        S["C2b"] = crit_turnover(runs, "c2b", "turnover", "perturb")
        S["C3b"] = crit_mix(runs, "c3b", "het_mixed", "het_isolated")
        S["C3b_N8"] = crit_mix(runs, "c3b", "het8_mixed", "het8_isolated")
        S["C3b_homog"] = crit_mix(runs, "c3b", "homog_mixed", "homog_isolated")
        S["C5b"] = crit_c5b(runs)
        scored = ["C1", "C2", "C2b", "C3b", "C4", "C5b"]
    else:
        S["C2"] = crit_turnover(runs, "c2", "turnover", "no_turnover")
        S["C3"] = crit_mix(runs, "c3", "mixed", "isolated")
        S["C5"] = crit_c5(runs)
        scored = ["C1", "C2", "C3", "C4", "C5"]

    plt, made = _plt(), []
    tex = "c2b" if v02 else "c2"
    fig_turnover(plt, runs.get(tex, {}),
                 [("turnover", S1, "Turnover (reset)"), ("perturb", S3, "Perturbation (matched)"),
                  ("no_turnover", S2, "No turnover")],
                 f"{'C2/C2b' if v02 else 'C2'} · Held-out accuracy (mean; band = min–max over seeds)",
                 results_dir, f"{tex}_curves.png", made)
    if v02:
        g = runs.get("c3b", {})
        groups = [(lab, [("Mixed (all pairs)", final(g[f"{k}_mixed"]), S1),
                         ("Isolated (own pair)", final(g[f"{k}_isolated"]), S2)])
                  for k, lab in (("het", "Heterogeneous N=4"), ("het8", "Heterogeneous N=8"), ("homog", "Homogeneous N=4"))
                  if g.get(f"{k}_mixed") and g.get(f"{k}_isolated")]
        if groups:
            fig_bars(plt, groups, "C3b · Held-out accuracy at equal compute (dots = seeds)", results_dir, "c3b_bars.png", made)
        if S["C5b"]:
            fig_sweep(plt, S["C5b"], "c5b", results_dir, made)
    else:
        g = runs.get("c3", {})
        if g.get("mixed") and g.get("isolated"):
            fig_bars(plt, [("N=4", [("Mixed (all pairs)", final(g["mixed"]), S1),
                                    ("Isolated (own pair)", final(g["isolated"]), S2)])],
                     "C3 · Held-out accuracy at equal compute (dots = seeds)", results_dir, "c3_bars.png", made)
        if S["C5"]:
            fig_sweep(plt, S["C5"], "c5", results_dir, made)

    any_run = next((r for e in runs.values() for rs in e.values() for r in rs), None)
    L = [f"# EI-Bench {'v0.2' if v02 else 'v0'} — results", ""]
    if any_run:
        n = sum(len(rs) for e in runs.values() for rs in e.values())
        wall = sum(r["wall_s"] for e in runs.values() for rs in e.values() for r in rs)
        L += [f"Runs: **{n}** · summed train time {wall / 60:.1f} min · device: `{any_run['device_name']}` · torch {any_run['torch']}", ""]
    L += ["## EI profile", "", "| Criterion | Verdict | Key number |", "|---|---|---|"]

    def key(k, s):
        if k == "C1":
            return f"held-out acc {fmt_ci(s['heldout_acc'])}, ratio {s['heldout_over_train']:.2f}"
        if k in ("C2", "C2b"):
            return f"Δ({s['arms'][0]}) − Δ({s['arms'][1]}) = {fmt_ci(s['diff'])}; MW p={s['mannwhitney_p_delta']:.3g}"
        if k in ("C3", "C3b"):
            return f"mixed − isolated = {fmt_ci(s['diff'])}; MW p={s['mannwhitney_p']:.3g}"
        if k == "C4":
            return f"topsim gain {fmt_ci(s['topsim']['diff'])}; posdis gain {fmt_ci(s['posdis']['diff'])}"
        if k == "C5":
            return f"J(exact) {s['J_exact']:.2f}, J(per-attr) {s['J_per_attr']:.2f}"
        if k == "C5b":
            return f"R = w_c/w_e = {s['R']:.2f} [{s['R_ci'][0]:.2f}, {s['R_ci'][1]:.2f}]"
    for k in scored:
        s = S.get(k)
        L.append(f"| {k} {NAMES[k]} | **{s['verdict']}** | {key(k, s)} |" if s else f"| {k} {NAMES[k]} | NOT RUN | |")
    L += ["", "Values: mean [95% bootstrap CI over seeds]; MW = two-sided Mann–Whitney U (secondary). "
          "Thresholds are pre-registered in DEFINITION.md.", ""]

    if S["C1"]:
        s = S["C1"]
        L += ["## C1 · Beyond the objective", "",
              f"- Train accuracy: {fmt_ci(s['train_acc'])}",
              f"- Held-out accuracy (combinations never trained on): {fmt_ci(s['heldout_acc'])}  (chance = {s['chance_exact']:.2e})",
              f"- Held-out / train ratio: {s['heldout_over_train']:.2f}",
              f"- Per seed: {', '.join(f'{v:.3f}' for v in s['per_seed'])}", ""]
    if S["C4"]:
        s = S["C4"]
        L += ["## C4 · Self-organized structure", "", "| Metric | Untrained | Trained | Gain |", "|---|---|---|---|"]
        for k in ("topsim", "posdis", "bosdis"):
            L.append(f"| {k} | {fmt_ci(s[k]['untrained'])} | {fmt_ci(s[k]['trained'])} | {fmt_ci(s[k]['diff'])} |")
        L.append("")
    for k in ("C2", "C2b"):
        s = S.get(k)
        if not s:
            continue
        a, b = s["arms"]
        L += [f"## {k} · {NAMES[k]}", ""]
        if k == "C2":
            L += [f"![C2]({tex}_curves.png)", ""]
        L += [f"- Gain after turnover start — {a}: {fmt_ci(s['delta_a'])}; {b}: {fmt_ci(s['delta_b'])}",
              f"- Final held-out accuracy — {a}: {fmt_ci(s['final_a'])}; {b}: {fmt_ci(s['final_b'])} (MW p={s['mannwhitney_p_final']:.3g})",
              f"- Topsim gain difference: {fmt_ci(s['topsim_delta_diff'])}"]
        for arm, vals in s["per_seed_final"].items():
            L.append(f"- Per-seed final ({arm}): {', '.join(f'{v:.3f}' for v in vals)}")
        L.append("")
    for k, lab in (("C3", "C3 · Interaction as source"), ("C3b", "C3b · Heterogeneous N=4 (scored)"),
                   ("C3b_N8", "C3b · Heterogeneous N=8 (secondary)"), ("C3b_homog", "C3b · Homogeneous N=4 replicate (secondary)")):
        s = S.get(k)
        if not s:
            continue
        L += [f"## {lab}", ""]
        if k in ("C3", "C3b"):
            L += [f"![C3]({'c3b' if v02 else 'c3'}_bars.png)", ""]
        mc, ic = list(s["cross_play"])
        L += [f"- Mixed (all N² pairs): {fmt_ci(s['mixed'])}; isolated (own pair): {fmt_ci(s['isolated'])}",
              f"- Difference: {fmt_ci(s['diff'])}; MW p={s['mannwhitney_p']:.3g}",
              f"- Cross-play: mixed {fmt_ci(s['cross_play'][mc])}, isolated {fmt_ci(s['cross_play'][ic])}",
              "- Structure (mixed − isolated): " + ", ".join(f"{m} {fmt_ci(v)}" for m, v in s["secondary_diff"].items()), ""]
    for k in ("C5", "C5b"):
        s = S.get(k)
        if not s:
            continue
        e = k.lower()
        L += [f"## {k} · {NAMES[k]}", "", f"![{k}]({e}_sweep.png)", "", f"![{k} NLL]({e}_nll.png)", "",
              "| Hidden | Seeds | Exact | Per-attribute | NLL |", "|---|---|---|---|---|"]
        for r in s["rows"]:
            L.append(f"| {r['hidden']} | {r['n']} | {r['exact']:.3f} | {r['per_attr']:.3f} | {r['nll']:.3f} |")
        if k == "C5b":
            L += ["", f"Transition width (10–90%, octaves): exact {s['w_exact']:.2f} [{s['w_exact_ci'][0]:.2f}, {s['w_exact_ci'][1]:.2f}], "
                  f"per-attribute {s['w_per_attr']:.2f} [{s['w_per_attr_ci'][0]:.2f}, {s['w_per_attr_ci'][1]:.2f}]; "
                  f"R = {s['R']:.2f} [{s['R_ci'][0]:.2f}, {s['R_ci'][1]:.2f}]. "
                  f"(Unscored v0 statistic: J(exact) {s['J_exact']:.2f}, J(per-attr) {s['J_per_attr']:.2f}.)", ""]
        else:
            L += ["", f"J(exact) = {s['J_exact']:.2f}, J(per-attribute) = {s['J_per_attr']:.2f}, J(−NLL) = {s['J_neg_nll']:.2f}.", ""]

    with open(os.path.join(results_dir, "REPORT.md"), "w") as f:
        f.write("\n".join(L))
    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(S, f, indent=1)
    return S, made, scored


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results/v02")
    a = p.parse_args(argv)
    S, figs, scored = build(a.results)
    for k in scored:
        print(f"{k}: {S[k]['verdict'] if S.get(k) else 'NOT RUN'}")
    print(f"wrote {a.results}/REPORT.md, summary.json and {len(figs)} figures")


if __name__ == "__main__":
    main()
