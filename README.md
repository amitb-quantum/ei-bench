# EI-Bench

**Emergent Intelligence, defined so it can be tested and can fail.**

*Quantum Clarity LLC · AB · started 2026-10-29 · current version v0.3 (pre-registered)*

---

## Why this exists

Everyone says "AI." For today's systems, "artificial" names only *where* the intelligence comes from:
it was built. It says nothing about *how* their abilities arose. In large models, most
abilities were never written down by anyone. They came out of scale, data, and interaction.

"Emergent intelligence" would be a better name for that, but only if it means something
specific. A rebrand isn't worth much. A definition is worth something if it is **operational**
(it can be measured), **discriminating** (some systems pass and some fail), and
**honest about failure** (a FAIL counts as a result).

EI-Bench is that definition plus a harness that measures it.

## The definition in one table

A system **S** is built from a designer specification **D** (architecture, objective, data,
training). A capability of S is *emergent with respect to D* to the degree that it meets five
criteria. We report an **EI profile**, one verdict per criterion, rather than one yes/no label.

| # | Criterion | The question |
|---|---|---|
| **C1** | Beyond the objective | Does S do something D never rewarded? |
| **C2** | Ongoing emergence | Does S keep developing after convergence, beyond what more of the same training gives? |
| **C3** | Interaction as source | Is part of the capability caused by interaction between agents or components, with compute held equal? |
| **C4** | Self-organized structure | Does unspecified structure form, measurably above an untrained baseline? |
| **C5** | Jumps that survive smooth metrics | If a capability seems to "switch on" with scale, does the jump survive continuous metrics? (the "mirage" test) |

Each verdict is **PASS**, **FAIL** or **INCONCLUSIVE**. Every threshold is pre-registered
in [`DEFINITION.md`](DEFINITION.md) before the experiments that test it are run.

## The current testbed: agents that invent a language

v0 / v0.2 use a deliberately small, well-studied setting, so the measurement harness
can be validated before it is pointed at anything ambitious.

- **Objects** have 4 attributes with 8 values each (4,096 objects). **10% of combinations are
  held out** and never seen in training.
- A **sender** describes an object with a 6-symbol message from a 16-symbol vocabulary. A
  **receiver** must reconstruct the object from the message alone. Nothing specifies the
  language, its grammar, or what any symbol means.
- **Populations** of senders and receivers either talk to random partners each step (*mixed*)
  or stay in fixed pairs (*isolated*), at the same compute.
- **Turnover** replaces an agent with a newcomer, who must learn the language from the others
  (iterated learning). v0.2 adds a control that perturbs an agent's weights by the same
  amount instead of replacing it.
- Agents may be **heterogeneous** (each with its own learning rate), and model size is
  swept to look for capability "jumps."

## Results so far

### v0 (2026-10-29 · 46 runs · 5 seeds · RTX 5090)

| # | Verdict | Key number |
|---|---|---|
| C1 | **PASS** | held-out accuracy 0.993 [0.981, 1.000] on combinations never trained on |
| C2 | **PASS** | turnover's gain exceeds the control's by 0.108 [0.032, 0.210]; all 5 turnover seeds ≥ 0.997, control seeds 0.656–0.990 |
| C3 | **FAIL** | mixed − isolated = −0.020 [−0.053, 0.010] |
| C4 | **INCONCLUSIVE** | topsim gain clears its threshold; posdis gain misses the 0.05 bound; bosdis shows nothing |
| C5 | **INCONCLUSIVE** | exact match looks like a jump; per-attribute accuracy rises steadily (leans "mirage") |

**What v0 showed:**
1. **Near-perfect generalization without much compositionality.** Agents get 99% of unseen
   combinations right using languages that are only weakly systematic (topsim ≈ 0.23). This
   matches Chaabouni et al. (2020): a language can work very well without being tidy.
2. **The apparent capability jump looks mostly like a metric effect.** Exact match jumps from
   0.02 to 0.99 across a few model sizes, while per-attribute accuracy and NLL improve steadily.
   This is the argument of Schaeffer et al. (2023), seen in our own data.
3. **Turnover reliably produces better languages, but the reason is still open.** It could
   be iterated learning (languages adapt to be learnable), or just disruption breaking a bad
   convention. v0.2 is designed to tell these apart.
4. **The harness can fail things.** C3 failed. A definition that every system passes would be useless.

### v0.2 (2026-10-30 · 160 runs · 10 fresh seeds · RTX 5090)

Pre-registered in `DEFINITION.md` §7 and committed before any v0.2 result existed. Results are in §8.

| Test | Verdict | Key number |
|---|---|---|
| C1 replication | **PASS** | held-out accuracy 0.970 [0.941, 0.993] |
| C2 replication | **PASS** | turnover − none = 0.076 [0.033, 0.133]; Mann–Whitney p = 0.001 |
| C2b reset vs. matched perturbation | **PASS** (with caveat) | reset − perturb = 0.896 [0.813, 0.963] |
| C3b heterogeneous populations | **FAIL** | mixed − isolated = −0.012 [−0.221, 0.135] |
| C4 replication | **INCONCLUSIVE** | topsim gain 0.152 [0.131, 0.171]; posdis gain just misses its bound |
| C5b transition width | **MIRAGE** | per-attribute transition 2.63× wider than exact match [1.99, 3.80] |

**What v0.2 showed:**
1. **The replications hold** with 10 independent seeds.
2. **The capability "jump" is a metric mirage.** Exact match appears to switch on between
   hidden sizes 32 and 96, while per-attribute accuracy improves steadily across about 3.8
   octaves. This reproduces Schaeffer et al. (2023) in a controlled system.
3. **Interaction adds no capability here, even with heterogeneous agents.** Populations
   agree on a shared language but do not generalize better. At N=8, mixing makes languages
   less compositional.
4. **Turnover robustly helps, and the mechanism is half-identified.** Replacing agents beats
   an equal-sized weight perturbation decisively. A post-hoc check showed, however, that the
   perturbation also destroys the agent's ability to learn (its weights grow, its outputs
   saturate), while a reset restores it. Separating "learning from others" from "restored
   plasticity" is the v0.3 question.

### v0.3 (pre-registered 2026-10-30 · 50 runs · fresh seeds from 200)

One question: does turnover help because a newcomer **relearns the language from the population**,
or only because a fresh agent **can learn again**? The control is **shrink-and-perturb**: the
agent's weights are scaled down to a fresh agent's size and lightly noised, which restores
plasticity but keeps latent knowledge. A calibration run before any experiment showed it
recovers about 8× faster than a reset, so the two arms differ in what the agent still knows,
not in weight scale. Also: C4 with 20 seeds to settle the posdis bound. Design and verdict rules
are in `DEFINITION.md` §9.

## Quickstart

Tested on WSL2 (Ubuntu 26.04) with an RTX 5090 and a conda/miniforge env. It also runs on CPU, just slower.

```bash
bash scripts/setup_env.sh            # creates conda env "eibench", installs, checks the GPU
conda activate eibench
pytest -q                            # 10 unit tests

bash scripts/launch.sh 3 --quick     # smoke test, a few minutes -> results/v03_quick
bash scripts/launch.sh 6             # full v0.3 grid, about 1.5-2 h on an RTX 5090 -> results/v03
python -m eibench.run --progress     # done / running per experiment
python -m eibench.report --results results/v03   # REPORT.md, summary.json, figures
```

- Workers restart themselves after a crash (for example a transient CUDA driver fault); see
  `grep -h crashed logs/worker_*.log`. Stop everything with `pkill -f scripts/worker.sh; pkill -f eibench.run`.
- `--suite v0` or `--suite v0.2` reruns an earlier grid. `--exp c2b,c5b` selects experiments, `--dry-run` lists
  pending runs, and `--device cpu` forces CPU.
- Any number of workers can share a grid: runs are claimed with lock files, finished runs are
  skipped, and locks left by dead workers are reclaimed. Re-running a command resumes the grid.

## Methodology

- **Pre-registration.** Thresholds and grids go into `DEFINITION.md` and git *before* the runs
  that test them. Past verdicts are never re-scored, and follow-up questions get new, separately
  named tests (C2 → C2b).
- **Seeds are the unit of replication.** Verdicts use 95% bootstrap CIs over seeds. A
  Mann–Whitney U test is reported as a secondary statistic that is robust to outlier seeds.
- **Conservative controls.** Comparisons hold compute equal, and the C3 metric deliberately
  favors the isolated control.
- **Failure is reported.** FAIL and INCONCLUSIVE verdicts are published with the same
  prominence as PASS.

## Implementation notes

- **PyTorch**, chosen because official TensorFlow wheels don't yet ship RTX 50-series
  (sm_120) kernels; PyTorch has supported Blackwell since 2.7 with CUDA 12.8.
  `scripts/env_check.py` verifies this.
- **Stacked-parameter populations.** Every weight has a leading agent dimension, and forward
  passes use `torch.baddbmm`, so a whole population trains as one set of batched kernels.
  Turnover resets one agent's slice, including its Adam state. Perturbation adds noise of the
  same size to that slice. Heterogeneous learning rates scale each agent's update.
- **Metrics:** held-out exact and per-attribute accuracy, NLL, cross-play accuracy; topsim, posdis
  and bosdis for language structure.
- The models are tiny, so a single run can't fill a large GPU, which is why parallel workers
  are the default.

```
eibench/   config · data · agents · metrics · stats · train · experiments · run · report
scripts/   setup_env.sh · env_check.py · launch.sh · worker.sh (self-restarting worker)
tests/     split coverage · metric sanity · reset/perturb/shrink matching · per-agent LR · transition width
results/   (git-ignored) per-run JSON, REPORT.md, summary.json, figures
DEFINITION.md   the definition, thresholds, results log and pre-registrations
CHANGES.md      version history
```

## Scope

**In scope now:** a validated, reproducible measurement harness for the five criteria on a
small system where results can be checked against published literature.

**Not claimed:** that these agents are "intelligent," or that a PASS here would carry over to
other systems. v0's "beyond the objective" is the weak form: generalizing to unseen
combinations of a familiar kind.

## Roadmap

| Stage | What | Headline question |
|---|---|---|
| **v0.2** *(done)* | perturbation control, heterogeneous populations, mirage test | C5 is a mirage; heterogeneity does not rescue C3; turnover beats perturbation |
| **v0.3** *(pre-registered)* | shrink-and-perturb control; C4 with 20 seeds; then write up the toy-system EI profile | Is turnover's gain learning from others, or restored plasticity? |
| **v1** | **LLM agent society**: small open models with persistent memory, same harness and criteria, tasks that need coordination no single agent was trained for | Does the profile change when the agents are language models? |
| **v2** | continual-learning agents (weights updated in deployment) plus world-model grounding | C2 as the headline: development that never stops |
| **Side track** | physical substrates, such as MBQC (resource state plus adaptive measurement) | Emergence without a program |

Planned outputs: a short write-up of the v0/v0.2 EI profile, an expanded position paper on the
definition, and a public benchmark release once v1 runs.

## References

- Anderson, P. W. (1972). More is different. *Science.*
- Lewis, D. (1969). *Convention.*
- Kirby, S., Cornish, H. & Smith, K. (2008). Cumulative cultural evolution in the laboratory. *PNAS.*
- Chaabouni, R. et al. (2020). Compositionality and generalization in emergent languages. *ACL.*
- Wei, J. et al. (2022). Emergent abilities of large language models. *TMLR.*
- Rita, M. et al. (2022). On the role of population heterogeneity in emergent communication. *ICLR.*
- Schaeffer, R., Miranda, B. & Koyejo, S. (2023). Are emergent abilities of large language models a mirage? *NeurIPS.*

## Author

AB, Founder and Principal Investigator, **Quantum Clarity LLC** (Milpitas, CA) ·
[quantum-clarity.com](https://quantum-clarity.com) · GitHub [@amitb-gpu](https://github.com/amitb-gpu)

License: not yet chosen.
