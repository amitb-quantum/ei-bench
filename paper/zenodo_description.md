# Emergent Intelligence, Defined So It Can Fail: A Pre-registered Test in AI Agents That Invent Their Own Language

**Amit Brahmbhatt** · Quantum Clarity LLC, Milpitas, California, USA · ORCID: [0009-0003-5882-8476](https://orcid.org/0009-0003-5882-8476)

Preprint, version 1.0 (revised text) · 31 October 2026 · DOI: [10.5281/zenodo.23849135](https://doi.org/10.5281/zenodo.23849135)

## Abstract

“Emergent” is used loosely for capabilities that nobody designed. We propose an operational definition that can fail. A capability is emergent with respect to its designer specification to the degree that it meets five criteria: it goes beyond the training objective (C1), keeps developing after convergence (C2), is caused by interaction at equal compute (C3), comes with self-organized structure above an untrained baseline (C4), and, if it appears to switch on with scale, survives continuous metrics (C5). A system receives an *EI profile* of **PASS**/**FAIL**/**INCONCLUSIVE** verdicts rather than a label. We implement the criteria in EI-Bench, an open harness, and apply it to populations of neural agents that invent a discrete language in a reconstruction game. Every threshold was pre-registered in a public git history before the runs that test it. Across three rounds and 256 runs on one GPU, the profile is: C1 **PASS** (held-out accuracy 0.90–0.99 on unseen attribute combinations); C2 **PASS**, observed in all three rounds; C3 **FAIL**, including with heterogeneous populations; C4 **FAIL** at 20 seeds; and the apparent capability jump with model size is a metric **MIRAGE** (per-attribute transition 2.6× wider than the exact-match transition). Here C1 tests generalization to unseen combinations within the trained task. Turnover also outperformed a shrink-and-perturb control designed to restore plasticity while retaining learned weight directions: the difference in held-out accuracy gain was +0.19 (95% CI [0.004, 0.409]). This supports a role for intergenerational transmission, without uniquely identifying the mechanism. In this testbed, turnover improved generalization, whereas mixing partners did not demonstrate an advantage: improvement came from generations, not crowds. We report two places where our pre-registered wording overclaimed or needed tightening, and how we handled them.

## Plain-language summary

AI systems are built by people, but many of their abilities are learned rather than programmed. We asked whether claims that an ability “emerged” can be tested in a way that could also come out negative. We wrote down five checks and built small AI agents that must invent their own code words to describe objects to each other. Before each round of experiments, we published the rules for judging its results. The agents learned languages that also worked for combinations of objects they had never seen. An apparent sudden leap in ability as the agents got bigger turned out to be mostly an effect of strict scoring: with partial credit, the improvement was gradual. Mixing up which agents talked to each other did not make them better than keeping fixed partners. But regularly replacing agents with newcomers, who had to learn the language from the others, improved results in all three rounds, and did better than the “refresh” we tested as a comparison. That supports the idea that passing a language on to new learners can improve it, although the exact mechanism is still open. The languages showed some structure, but not enough to pass our structure test. In this testbed, improvement came from generations, not crowds. Next, we plan to apply these checks to teams of language-model agents.

## 1. Introduction

“Artificial intelligence” describes systems built by people; “emergence” concerns how capabilities arise within them. These descriptions are compatible. Learned abilities need not be explicitly programmed, but that alone does not distinguish emergence from ordinary learning and generalization. A useful account must specify what evidence supports the claim and what would count against it. Claims of abrupt “emergent abilities” of language models [8] can be artefacts of discontinuous metrics [7].

We therefore treat emergence as an empirical property with a falsifiable test. Our contributions:

1.  **An operational definition** with five criteria and explicit verdict rules, reported as a profile rather than a label (Section 3).

2.  **EI-Bench**, an open, single-GPU harness that implements the criteria for populations of communicating agents, with batched population training, turnover and perturbation controls, and bootstrap statistics (Section 4).

3.  **A pre-registered, three-round study** of 256 runs, with thresholds committed to git before each round (Section 5). The profile it yields is C1 **PASS**, C2 **PASS**, C3 **FAIL**, C4 **FAIL**, C5 **MIRAGE**.

4.  **A controlled result for C2**: generational turnover improves held-out performance across three rounds and outperforms the tested shrink-and-perturb control, supporting an intergenerational-transmission interpretation.

## 2. Related work

**Emergent communication.** Agents trained to coordinate through a discrete channel develop conventions in the sense of Lewis signalling games [5]. Chaabouni et al. [2] showed that such languages can generalize to unseen attribute combinations without being compositional, and introduced the positional and bag-of-symbols disentanglement measures we use. Rita et al. [6] found that population effects depend on heterogeneity among agents.

**Iterated learning.** When a language must be relearned by each new generation, transmission favours structure that is easy to learn [4]. Our turnover manipulation is a neural version of this idea.

**Emergent abilities and metrics.** Wei et al. [8] documented abilities that appear abruptly with scale; Schaeffer et al. [7] argued many are produced by non-linear or discontinuous metrics. Our C5 criterion operationalizes that critique.

**Plasticity.** Deep networks can lose the ability to learn under continued training [3]; re-initialization and shrink-and-perturb [1] can help restore it. This is an alternative explanation for turnover’s benefit. Our C2c comparison tests a specific plasticity-oriented control, rather than excluding all plasticity-related explanations.

## 3. An operational definition

Let a system S be produced by a designer specification D = (architecture, objective, data, training procedure). A capability of S is *emergent with respect to D* to the degree that it meets the criteria in Table 1. These are proposed operational criteria, not necessary and sufficient conditions for intelligence. In this testbed, C1 is a deliberately limited proxy: generalization to held-out combinations within the same reconstruction task.

| Criterion                       | Question                                                                                    |
|:--------------------------------|:--------------------------------------------------------------------------------------------|
| C1 Beyond the objective         | Does S do something D never rewarded?                                                   |
| C2 Ongoing emergence            | Does S keep developing after convergence, beyond more of the same training?               |
| C3 Interaction as source        | Is part of the capability caused by interaction among agents, at equal compute?             |
| C4 Self-organized structure     | Does unspecified structure form, above an untrained baseline?                               |
| C5 Jumps survive smooth metrics | If a capability seems to switch on with scale, does the jump hold under continuous metrics? |

**Table 1.** The five criteria. Each receives a verdict; together they form the system’s EI profile.

**Verdicts.** Seeds are the unit of replication. We report means with 95% percentile bootstrap confidence intervals over seeds (10,000 resamples). A test is **PASS** when the whole interval clears its threshold, **INCONCLUSIVE** when only the mean does, and **FAIL** otherwise; C5 adds **MIRAGE** and NO-JUMP outcomes. A two-sided Mann–Whitney U test is reported as a secondary rank-based statistic. A **FAIL** is a verdict under the specified rule, not proof that the underlying effect is absent. In particular, an interval spanning zero does not establish equivalence between conditions.

**Pre-registration.** Every round’s design, thresholds and verdict rules were written into `DEFINITION.md` and committed to a public git repository before any run of that round. Earlier verdicts are never re-scored; follow-up questions receive new, separately named tests (C2 → C2b → C2c).

## 4. Testbed and methods

**Task.** Objects have A = 4 attributes with V = 8 values each (4,096 objects). A sender maps an object to a message of L = 6 symbols from a vocabulary of K = 16; a receiver must reconstruct every attribute from the message alone. Ten percent of attribute combinations (410 objects) are held out from training, while every individual value still appears in training. Nothing in D specifies the language, its structure, or what any symbol means.

**Agents.** Senders and receivers are GRU networks (hidden size 128, embedding 32 unless swept). Symbols are discrete in the forward pass and trained with straight-through Gumbel-softmax (τ = 1). Populations use stacked parameters with a leading agent dimension, so a population of N senders and N receivers trains as one set of batched matrix multiplications.

**Training.** Adam (learning rate 10⁻³), batch 256, 8,000 steps, with linear learning-rate decay to 0.05× over the last 20% of steps. In *mixed* populations, sender–receiver pairings are re-drawn every step; in the *isolated* control, each sender keeps one receiver. Both use identical parameters and compute.

**Turnover (C2).** Mixed populations with N = 4 train for 24,000 steps. From step 8,000 until 85% of training, every 500 steps one randomly chosen agent (sender or receiver) is replaced by a freshly initialized agent and its Adam state is cleared (24 events per run). The learning rate decays over the final 15%. The C2 metric is the change in held-out accuracy from step 8,000 (the start of the turnover phase) to the end, compared against an arm without turnover.

**Controls.** *C2b* (v0.2) adds isotropic Gaussian noise whose norm matches the displacement a reset would cause. *C2c* (v0.3) applies shrink-and-perturb [1]: w′ = λ·w + σ·f, with f a fresh initialization sample, σ = 0.1 and λ = ‖f‖ / ‖w‖ per parameter tensor, so the agent returns to fresh-init scale while keeping the direction of its learned weights.

**Metrics.** C1: exact-match accuracy on held-out combinations (chance 2.4 × 10⁻⁴). C3: held-out accuracy, scored over all N² pairings for mixed populations and own pairs only for isolated ones, a comparison that deliberately favours the control. C4: topographic similarity (Spearman correlation of pairwise meaning and message Hamming distances, 1,000 objects), and positional and bag-of-symbols disentanglement [2], each compared with the same agents before training; a **PASS** needs topsim gain > 0.05 and posdis or bosdis gain > 0.05. C5: model size is swept and exact-match accuracy is compared with per-attribute accuracy (Section 5.3).

**Compute.** All runs used one NVIDIA RTX 5090 under WSL2 with PyTorch 2.15 (CUDA 13.2), with up to six worker processes sharing a lock-file work queue. Summed training time was 136 min (v0, 46 runs), 934 min (v0.2, 160 runs) and 531 min (v0.3, 50 runs).

## 5. Results

Table 2 summarizes all scored tests. Seeds were 0–4 (v0), 100–109 (v0.2) and 200–219 (v0.3), never reused.

| Round           | Test           | Verdict            | Key number (mean [95% CI])                                                      |
|:----------------|:---------------|:-------------------|:----------------------------------------------------------------------------------|
| v0 (5 seeds)    | C1             | **PASS**           | held-out acc. 0.993 [0.981, 1.000]                                              |
|                 | C2             | **PASS**           | Δ(turnover) − Δ(none) = 0.108 [0.032, 0.210]              |
|                 | C3             | **FAIL**           | mixed − isolated = −0.020 [−0.053, 0.010]                 |
|                 | C4             | **INCONCLUSIVE**   | posdis gain 0.070 [0.033, 0.105]                                                |
|                 | C5             | **INCONCLUSIVE**   | jump ratios 0.58 (exact), 0.31 (per-attribute)                                    |
| v0.2 (10 seeds) | C1             | **PASS**           | 0.970 [0.941, 0.993]                                                            |
|                 | C2             | **PASS**           | 0.076 [0.033, 0.133]; MW p = 0.001                                            |
|                 | C2b            | **PASS**† | Δ(reset) − Δ(perturb) = 0.896 [0.813, 0.963]              |
|                 | C3b            | **FAIL**           | heterogeneous: −0.012 [−0.221, 0.135]                       |
|                 | C4             | **INCONCLUSIVE**   | posdis gain 0.076 [0.040, 0.112]                                                |
|                 | C5b            | **MIRAGE**         | width ratio R = 2.63 [1.99, 3.80]                                             |
| v0.3            | C1 (20 seeds)  | **PASS**           | 0.899 [0.800, 0.975]                                                            |
|                 | C2 (10 seeds)  | **PASS**           | 0.233 [0.069, 0.443]; MW p = 0.017                                            |
|                 | C2c (10 seeds) | **PASS**           | Δ(reset) − Δ(S&P) = 0.190 [0.004, 0.409]; MW p = 0.045 |
|                 | C4 (20 seeds)  | **FAIL**           | posdis gain 0.039 [0.021, 0.058]                                                |

**Table 2.** All scored verdicts. †See Section 6 on what C2b can and cannot show.

### 5.1 C1: generalization beyond the objective

Single sender–receiver pairs reconstruct held-out combinations almost as well as trained ones (held-out/train ratio 0.95–0.99), at roughly 4,000 times chance. In v0.3, 2 of 20 pairs did not converge (0.195, 0.444), which lowers the mean but not the verdict. This is the weak form of C1: generalization to unseen combinations of familiar parts.

### 5.2 C4: structure is real but not disentangled

Topographic similarity rises reliably above the untrained baseline in every round (gain 0.114–0.152). Positional disentanglement improves too, but its confidence interval does not clear the pre-registered 0.05 margin: **INCONCLUSIVE** at 5 and 10 seeds, where the mean exceeds the margin, and **FAIL** at 20 seeds, where the mean is below it (gain 0.039 [0.021, 0.058]). The latter interval still spans the margin. Bag-of-symbols disentanglement shows no gain. Thus C4 does not pass its combined rule; this is not evidence that the languages have no structure. As in [2], strong generalization need not imply strong compositionality.

### 5.3 C5: the capability jump is a metric mirage

v0.2 swept 12 hidden sizes from 4 to 256 (5 seeds each). Exact-match accuracy appears to switch on between sizes 32 and 96 (Figure 1). Per-attribute accuracy, a graded measure of the same predictions, rises steadily over the whole range. The 10–90% transition widths are 1.44 octaves for exact match and 3.78 octaves for per-attribute accuracy, a ratio of 2.63 [1.99, 3.80], which meets the pre-registered **MIRAGE** rule (R ≥ 2 with the CI lower bound above 1.5). Because exact match requires all four attributes to be correct, it compresses a gradual improvement into an apparent jump, as argued in [7].

**Figure 1.** C5b. Held-out accuracy versus hidden size (mean of 5 seeds per size). The apparent jump in exact match (blue) corresponds to a gradual rise in per-attribute accuracy (orange). *(Figure shown in the PDF.)*

### 5.4 C3: partner mixing does not demonstrate a generalization advantage

Mixed populations reliably converge on a shared language: senders and receivers that never trained as a fixed pair understand each other (cross-play accuracy 0.82–0.96, against 0 for isolated pairs). This does not count towards C3, because the mixed objective requires it. On the capability that C3 scores, held-out generalization, mixed populations did not demonstrate an advantage over isolated pairs at equal compute. Point estimates were negative with homogeneous agents (v0: −0.020; v0.2 replicate: −0.016) and when agents learned at different speeds (v0.2 C3b, learning-rate multipliers from 0.5 to 2: −0.012 [−0.221, 0.135]). The heterogeneous interval permits both benefit and harm; the **FAIL** verdict does not establish equivalence. Both arms contain interacting sender–receiver pairs, so this tests partner mixing rather than interaction versus no interaction. At N = 8, mixing even made languages less disentangled (posdis −0.052 [−0.076, −0.026], secondary).

### 5.5 C2: turnover improves generalization; controls probe the mechanism

Replacing agents during training improved held-out accuracy relative to continued training in all three rounds (Table 2). The no-turnover arm tends to drift after convergence, and some seeds collapse (to 0.426 in v0.2 and 0.118 in v0.3), while turnover arms mostly end near 1.0.

**Is it disruption?** v0.2 compared turnover with a perturbation of matched size. Perturbed populations collapsed (final accuracy 0.066), which made C2b pass by a wide margin. A post-hoc diagnostic on a separately trained population showed why: matched noise left the perturbed agent’s weights about 1.5× larger than its trained weights, with an output layer about 9× larger than at initialization, so its symbol choices saturated and it largely stopped learning. C2b thus showed that re-initialization beats an equal-sized disruption, but not *why*.

**Is it restored plasticity?** v0.3 compared turnover with shrink-and-perturb, which returns the agent to fresh-init scale while keeping its learned direction (in the experiment runs, averaged over 240 events: weight norm 0.44 of the pre-intervention agent’s, cosine 0.945 to its old weights). A calibration run before any v0.3 experiment confirmed the intended contrast: shrink-and-perturb brought weight norm to 0.50 of the trained agent’s (0.49 for a reset); the agent’s accuracy with its partners fell to 0.02 but recovered to 0.53 within 50 steps, against 0.07 for a reset. Its knowledge was latent, not lost.

The result (Figure 2): turnover beat shrink-and-perturb by 0.190 [0.004, 0.409] (**PASS**, Mann–Whitney p = 0.045), while the difference between shrink-and-perturb and no turnover was not statistically significant (final accuracy 0.771 vs. 0.728, p = 0.79). This latter comparison does not establish equivalence. Turnover also produced more structured languages than shrink-and-perturb (topsim gain difference 0.046 [0.004, 0.093]). These findings support an intergenerational-transmission interpretation consistent with iterated learning [4]: turnover’s benefit exceeds that of the tested shrink-and-perturb intervention. However, matching weight scale and observing rapid recovery do not establish that the interventions restore plasticity identically. The comparison therefore narrows the mechanism without uniquely identifying it. C2c clears its pre-registered threshold, but its lower confidence bound is close to zero and the difference is partly driven by control seeds that collapse. The result may reflect improved stability as well as gains in performance; it does not establish indefinite or monotonic improvement.

**Figure 2.** C2c. Final held-out accuracy per seed (10 seeds per arm, N = 4, 24,000 steps). Turnover ends near 1.0; no statistically significant difference was detected between shrink-and-perturb and no turnover. *(Figure shown in the PDF.)*

## 6. Deviations, corrections and incidents

We list every departure from the pre-registered plan, because the method’s value depends on it.

1.  **C2b wording overclaimed.** The v0.2 pre-registration said a C2b **PASS** would mean “the gain comes from relearning the language from others”. The post-hoc saturation check showed C2b cannot support that reading. The verdict stands; the wording is retained in the record and explicitly corrected in the results (`DEFINITION.md` §8). The question was carried into v0.3.

2.  **C2c design was calibrated, and its reading tightened, before any run.** A first calibration showed that shrink-and-perturb destroys immediate function, which raised the question of whether it was a valid control. A recovery test showed knowledge is retained in latent form; this calibration was written into the pre-registration. The rule for reading an equivalence result was extended from “FAIL with the CI inside [−0.03, 0.03]” to any non-**PASS** with that CI, before any v0.3 run. A proposed “reset all agents” arm was dropped before running, with the reason recorded.

3.  **One worker crash.** In v0.2 one of six worker processes died from a transient CUDA driver error. Results are written only when a run completes, and a stale lock is reclaimed, so the interrupted run was redone in full by another worker; all 160 runs completed. Workers restart themselves automatically from v0.3 on.

## 7. Limitations

This is a deliberately small system. “Beyond the objective” here is compositional generalization, a weak form of C1. Results come from one task, one architecture family and one channel size, and C3 was tested only for pairings re-drawn every step. Thresholds such as the 0.05 disentanglement margin are conventions we fixed in advance, not natural constants. With 10 seeds per arm, C2c has limited precision; a larger replication would sharpen the effect size. The 256 runs span multiple tests, model sizes and controls; they are not 256 independent replications of each claim. A stronger mechanism test would vary access to established communication partners while controlling the reset intervention and training budget. Finally, the profile describes performance under these operational tests, not intelligence in general. Applying the framework to other systems requires suitable tasks, controls and pre-registered thresholds; their verdicts may differ.

## 8. Conclusion and future work

EI-Bench makes five proposed criteria for emergence operational and testable on a single GPU. In this communication testbed, it returned two passes, two failures and one metric mirage, with all outcomes reported. The clearest positive result is that generational turnover improved held-out performance relative to continued training across three rounds. Its advantage over the tested shrink-and-perturb control supports a transmission-based explanation, while leaving the mechanism open to stronger tests. Partner mixing did not demonstrate a generalization advantage at equal compute. Together, these results make generational turnover a promising direction for further study and show why emergence claims benefit from explicit controls and measurement checks. The next step is to adapt and pre-register the framework for teams of language-model agents with persistent memory, using a deterministic runtime authority layer to record tamper-evident evidence.

## Data and code availability

Code (Apache-2.0), the full pre-registration history and all reports are in EI-Bench, <https://github.com/amitb-quantum/ei-bench>. All 256 per-run result files, the scored reports and the pre-registration history are included as a dataset file in this Zenodo record.

## Revision note

Editorial clarification, 31 October 2026. This revised text corrects the experimental-round count (three rounds, not four) and clarifies the scope of generalization, the interpretation of benchmark failures, the scope of the C3 test (partner mixing) and the evidence for the turnover mechanism. Reported numerical results, thresholds and verdicts are unchanged. No new experiments are reported. The data file in this record is unchanged.

## Acknowledgements

Experiment design, code and drafting were carried out with the assistance of an AI system (Claude, Anthropic). All experiments were run, checked and approved by the author, who takes responsibility for the content.

## References

1. J. T. Ash and R. P. Adams. On warm-starting neural network training. *NeurIPS*, 2020.
2. R. Chaabouni, E. Kharitonov, D. Bouchacourt, E. Dupoux and M. Baroni. Compositionality and generalization in emergent languages. *ACL*, 2020.
3. S. Dohare, J. F. Hernandez-Garcia, Q. Lan, P. Rahman, A. R. Mahmood and R. S. Sutton. Loss of plasticity in deep continual learning. *Nature*, 632:768–774, 2024.
4. S. Kirby, H. Cornish and K. Smith. Cumulative cultural evolution in the laboratory: an experimental approach to the origins of structure in human language. *PNAS*, 105(31):10681–10686, 2008.
5. D. Lewis. *Convention: A Philosophical Study*. Harvard University Press, 1969.
6. M. Rita, F. Strub, J.-B. Grill, O. Pietquin and E. Dupoux. On the role of population heterogeneity in emergent communication. *ICLR*, 2022.
7. R. Schaeffer, B. Miranda and S. Koyejo. Are emergent abilities of large language models a mirage? *NeurIPS*, 2023.
8. J. Wei et al. Emergent abilities of large language models. *Transactions on Machine Learning Research*, 2022.
