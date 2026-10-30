# Emergent Intelligence — an operational definition (v0.2)

*Quantum Clarity LLC · draft for discussion · v0.1 2026-10-29 · v0.2 pre-registration added 2026-10-29 (§6–7) · v0.2 results added 2026-10-30 (§8)*

## 1. Why a definition, not a rename

"Artificial" names where intelligence comes from (it is built). "Emergent" should name
**how a capability arises**: from scale, interaction, or self-organization, rather than
from explicit specification. To be useful, the term has to separate systems, so it must be
testable and able to fail.

## 2. Definition

Let a system **S** be produced by a designer specification **D** = (architecture,
objective, data, training procedure). A capability **c** of S is *emergent with respect to D* to
the degree that it meets the criteria below. We report an **EI profile**, one verdict per
criterion, instead of a single yes/no. A system is not "emergently intelligent" in the
abstract; it has a measured profile.

| # | Criterion | Question it answers |
|---|-----------|---------------------|
| **C1** | **Beyond the objective** | Does S do something D never rewarded? |
| **C2** | **Ongoing emergence** | Does S keep developing *after* initial convergence, beyond what more of the same training gives? |
| **C3** | **Interaction as source** | Is some of the capability caused by interaction among components or agents, with compute held equal? |
| **C4** | **Self-organized structure** | Does unspecified internal or communicative structure form, measurably above an untrained baseline? |
| **C5** | **Jumps that survive smooth metrics** | If a capability seems to "switch on" with scale, does the jump hold up under continuous metrics (the "mirage" test, Schaeffer et al. 2023)? |

Verdicts are **PASS**, **FAIL**, or **INCONCLUSIVE** (the effect points the right way but the
95% CI includes zero). Thresholds are fixed in this document *before* any run, i.e. pre-registered.

## 3. v0 testbed: a population that invents a language

The **reconstruction game** (after Lewis 1969; Chaabouni et al. 2020):

* **Objects** have A = 4 attributes, each with V = 8 values (4,096 objects).
* A **sender** sees an object and emits a fixed-length message of L = 6 symbols from a
  vocabulary of K = 16. Symbols are discrete, trained with straight-through Gumbel-softmax.
* A **receiver** sees only the message and must reconstruct every attribute.
* **10% of attribute combinations are held out** and never shown in training. Every
  individual value still appears in training.
* **Populations**: N senders and N receivers. In the *mixed* condition, pairings are
  re-shuffled every step. In the *isolated* control, each sender always talks to the same
  receiver. Both conditions use the same parameters and compute.
* **Turnover**: in a mixed population, a randomly chosen agent is periodically
  re-initialized and must learn the language from the others (iterated learning).

Nothing in D specifies the language, its structure, or what any symbol means.

### Operationalization and pre-registered thresholds

Seeds are the unit of replication. CI = 95% percentile bootstrap over seeds (10,000 resamples).

| # | Experiment | Primary metric | PASS if |
|---|-----------|----------------|---------|
| C1 | Single pair, S seeds | Exact-match accuracy on **held-out** combinations | CI lower bound of held-out accuracy ≥ 0.25 (~1,000× chance) **and** mean held-out/train ratio ≥ 0.5 |
| C2 | Mixed N=4, turnover vs. none, same steps | Δ = held-out accuracy at end − held-out accuracy when turnover starts | CI of (Δ_turnover − Δ_none) lies entirely > 0 |
| C3 | Mixed N=4 vs. isolated N=4, same steps | Held-out accuracy (mixed: mean over **all** N² sender–receiver pairs; isolated: own pair only) | CI of (mixed − isolated) lies entirely > 0 |
| C4 | Runs from C1 | Topographic similarity (topsim); positional (posdis) and bag-of-symbols (bosdis) disentanglement | CI of (trained − untrained) topsim > 0.05 **and** the same for posdis *or* bosdis |
| C5 | Single pair, hidden size swept | Jump ratio J = largest single-step rise in the normalized curve ÷ total rise | Exact-match J ≥ 0.5 **and** continuous-metric J ≥ 0.5 (per-attribute accuracy). If exact-match J ≥ 0.5 but continuous J < 0.3, the jump is labelled **MIRAGE** (FAIL). If exact-match J < 0.5, there is no jump to test (**NO-JUMP**, reported but not scored). |

Secondary metrics (reported, not scored): cross-play accuracy, train accuracy, held-out
negative log-likelihood, and training curves.

The C3 comparison deliberately favors the isolated control: each isolated pair is scored
only on its own private language, while the mixed population must work across every pairing.

## 4. Honest limitations of v0

* This is a **toy**. "Beyond the objective" here means compositional generalization
  to unseen combinations, a weak form of C1. v0 checks that the measurement harness
  works and can return FAIL. It does not show that anything is "intelligent."
* The literature is mixed on several of these effects. Chaabouni et al. (2020) found
  that generalization does not require compositionality, and population effects depend on
  heterogeneity (Rita et al. 2022). **A FAIL is a valid and publishable result.**
* Five seeds give low statistical power, so expect some INCONCLUSIVE verdicts.

## 5. Roadmap

* **v1**: the same harness and criteria with an **LLM agent society** (small open models
  with persistent memory), with tasks that need coordination no single agent is trained for.
* **v2**: continual-learning agents (weights updated in deployment) plus world-model
  grounding; C2 becomes the headline criterion.
* **Side track**: physical substrates (MBQC-style resource state plus adaptive measurement)
  as an "emergence without a program" case study.

## 6. v0 results (RTX 5090, 46 runs, 5 seeds)

| # | Verdict | Key number |
|---|---|---|
| C1 | PASS | held-out accuracy 0.993 [0.981, 1.000]; held-out/train 0.99 |
| C2 | PASS | Δ(turnover) − Δ(none) = 0.108 [0.032, 0.210]; all 5 turnover seeds ≥ 0.997, control seeds 0.656–0.990 |
| C3 | FAIL | mixed − isolated = −0.020 [−0.053, 0.010]; mixed cross-play 0.958 vs. 0 (required by the objective, so not counted) |
| C4 | INCONCLUSIVE | topsim gain 0.114 [0.061, 0.177] passes; posdis gain 0.070 [0.033, 0.105] misses the 0.05 bound; bosdis ≈ 0 |
| C5 | INCONCLUSIVE | J(exact) 0.58, J(per-attribute) 0.31; per-attribute accuracy and NLL rise steadily across sizes 16–128 |

Readings: (a) near-perfect generalization with only weakly compositional languages, as in
Chaabouni et al. (2020); (b) the C5 exact-match "jump" looks largely like a metric effect;
(c) the C2 mechanism is ambiguous: turnover might help through iterated learning, or just by
perturbing a population stuck in a poor convention.

## 7. v0.2 pre-registration (written after v0, before any v0.2 run)

v0 verdicts are final and are not re-scored. v0.2 adds new tests and an independent
replication with **10 fresh seeds (100–109)** and the same task and channel as v0. Results go
to `results/v02`.

| Test | Design | Primary metric | PASS if |
|---|---|---|---|
| **C1/C4-rep** | single pair, 10 seeds | as in v0 | as in v0 |
| **C2-rep** | mixed N=4: turnover (reset) vs. none, 24k steps | as in v0 | as in v0 |
| **C2b** perturbation control | adds a third arm, **perturb**: on the same schedule, the chosen agent's weights get isotropic Gaussian noise whose norm equals the displacement a reset would have caused (per parameter tensor), and its Adam state is cleared as in a reset | Δ held-out accuracy, as in C2 | CI of (Δ_reset − Δ_perturb) > 0, meaning the gain comes from relearning the language from others, not from disruption alone |
| **C3b** heterogeneous populations | N=4; each agent gets its own learning-rate multiplier, geometrically spaced in [0.5, 2] (senders ascending, receivers descending); applied identically to the mixed and isolated arms | held-out accuracy (scored as in C3) | CI of (het-mixed − het-isolated) > 0 |
| C3b secondary | same at N=8; homogeneous N=4 replicate | reported, not scored | — |
| **C5b** transition width | hidden ∈ {4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256} × 5 seeds | 10–90% transition width, in octaves of hidden size, of the seed-mean curve normalized to its own min/max; *w_e* (exact match) and *w_c* (per-attribute accuracy); ratio *R = w_c / w_e*; 95% CI by resampling seeds within each size | **NO-JUMP** if w_e > 2.5 octaves. Otherwise **MIRAGE** (FAIL) if R ≥ 2 and CI lower bound > 1.5; **PASS** if R ≤ 1.5 and CI upper bound < 2; else **INCONCLUSIVE** |

Secondary statistics for C2/C2b/C3b: a two-sided Mann–Whitney U test on the per-seed values,
reported but not scored. It is robust to single outlier seeds such as v0's control seed 3.

v0's J statistic is still reported for C5b for continuity, but it is not scored there,
because J depends on grid spacing and a finer grid lowers J by construction.

For reference, applying the C5b rule to v0's coarser grid gives w_e ≈ 1.6 and w_c ≈ 3.3
octaves (R ≈ 2.0). That calculation is the reason for the cut points above; v0.2 is the actual test.

## 8. v0.2 results (RTX 5090, 160 runs, seeds 100–109, completed 2026-10-30)

| Test | Verdict | Key number |
|---|---|---|
| C1-rep | PASS | held-out accuracy 0.970 [0.941, 0.993]; held-out/train 0.99 |
| C2-rep | PASS | Δ(turnover) − Δ(none) = 0.076 [0.033, 0.133]; MW p = 0.001 |
| C2b | PASS | Δ(reset) − Δ(perturb) = 0.896 [0.813, 0.963]; perturb arm final accuracy 0.066 |
| C3b | FAIL | het-mixed − het-isolated = −0.012 [−0.221, 0.135] |
| C4-rep | INCONCLUSIVE | topsim gain 0.152 [0.131, 0.171] passes; posdis gain 0.076 [0.040, 0.112] misses the 0.05 bound again |
| C5b | MIRAGE | transition width exact 1.44 vs. per-attribute 3.78 octaves; R = 2.63 [1.99, 3.80] |

**Readings.**

1. **Replications hold.** C1, C2 and C4 reproduce v0 with 10 independent seeds. The no-turnover
   control again drifts down after convergence (Δ = −0.040), with one seed collapsing to 0.426.
2. **The capability jump is a metric effect (C5b).** Exact-match accuracy appears to switch on
   between hidden sizes 32 and 96. Per-attribute accuracy and NLL improve steadily across about
   3.8 octaves. This reproduces Schaeffer et al. (2023) in a controlled toy system.
3. **Interaction adds no capability here, even with heterogeneity (C3b).** Mixed populations
   develop a shared language (cross-play 0.82–0.95 vs. 0), but no better generalization. At N=8,
   mixing makes languages *less* compositional (posdis −0.052 [−0.076, −0.026], secondary).
4. **C2b passes, but the control was not the clean comparison intended.** A post-hoc check
   (not pre-registered) shows that noise matched to the reset displacement leaves an agent with
   weight norms about 1.5× its trained values. Its output layer ends about 9× larger than a fresh
   one, so symbol choice is nearly fixed and gradients largely vanish. The perturbed agent
   loses plasticity; a reset agent regains it. So C2b shows that re-initialization beats an
   equal-sized disruption. It does **not** yet separate "relearning from the population" from
   "restored plasticity" (compare continual-learning results on resets and shrink-and-perturb).
   The §7 phrase "the gain comes from relearning the language from others" overstated what C2b
   can show. It is kept as written, as part of the pre-registration record.

**Carried to v0.3 (to be pre-registered before running):** a plasticity-restoring control
(shrink-and-perturb: w ← λw + small noise, matched so the result has fresh-init norm), and a
"reset without teachers" arm (all agents reset together) to isolate learning from others.

## References

* Lewis, D. (1969). *Convention.*
* Anderson, P. W. (1972). More is different. *Science.*
* Kirby, S. et al. (2008). Cumulative cultural evolution in the laboratory. *PNAS.*
* Chaabouni, R. et al. (2020). Compositionality and generalization in emergent languages. *ACL.*
* Wei, J. et al. (2022). Emergent abilities of large language models. *TMLR.*
* Rita, M. et al. (2022). On the role of population heterogeneity in emergent communication. *ICLR.*
* Schaeffer, R. et al. (2023). Are emergent abilities of large language models a mirage? *NeurIPS.*
