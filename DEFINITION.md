# Emergent Intelligence — an operational definition (v0.1)

*Quantum Clarity LLC · draft for discussion · 2026-10-29*

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

## References

* Lewis, D. (1969). *Convention.*
* Anderson, P. W. (1972). More is different. *Science.*
* Kirby, S. et al. (2008). Cumulative cultural evolution in the laboratory. *PNAS.*
* Chaabouni, R. et al. (2020). Compositionality and generalization in emergent languages. *ACL.*
* Wei, J. et al. (2022). Emergent abilities of large language models. *TMLR.*
* Rita, M. et al. (2022). On the role of population heterogeneity in emergent communication. *ICLR.*
* Schaeffer, R. et al. (2023). Are emergent abilities of large language models a mirage? *NeurIPS.*
