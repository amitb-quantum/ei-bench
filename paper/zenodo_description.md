## Plain-language summary

People call today's AI "artificial", but many of its abilities were never programmed by anyone; they appeared on their own. We asked whether that kind of "emergence" can be tested honestly, in a way that could also come out negative.

We wrote down five checks and built small AI agents that must invent their own code words to describe objects to each other. Before every experiment we published our pass/fail rules, like sealing a prediction in an envelope before the result is known.

What we found:

- **The agents invented working languages** that correctly describe objects they had never seen.
- **An apparent sudden leap in ability** as the agents got bigger turned out to be an illusion of how we scored them.
- **Putting more agents in conversation did not make them better.**
- **Regularly replacing agents with newcomers**, who had to learn the language from the others, steadily improved it, much as human languages are refined as each generation learns them. Giving old agents a "refresh" without that relearning did not help.

In short: in this system, emergence came from **generations, not crowds**. The same checks can now be applied to AI systems in use today, such as teams of AI assistants working together.

## Abstract

"Emergent" is used loosely for capabilities that nobody designed. We propose an operational definition that can fail. A capability is emergent with respect to its designer specification to the degree that it meets five criteria: it goes beyond the training objective (C1), keeps developing after convergence (C2), is caused by interaction at equal compute (C3), comes with self-organized structure above an untrained baseline (C4), and, if it appears to switch on with scale, survives continuous metrics (C5). A system receives an *EI profile* of PASS / FAIL / INCONCLUSIVE verdicts rather than a label.

We implement the criteria in EI-Bench, an open harness, and apply it to populations of neural agents that invent a discrete language in a reconstruction game. Every threshold was pre-registered in a public git history before the runs that test it. Across four rounds and 256 runs on one GPU, the profile is: C1 PASS (held-out accuracy 0.90–0.99 on unseen attribute combinations); C2 PASS, replicated three times; C3 FAIL, including with heterogeneous populations; C4 FAIL at 20 seeds; and the apparent capability jump with model size is a metric MIRAGE (per-attribute transition 2.6× wider than the exact-match transition).

A shrink-and-perturb control shows that the turnover effect is not explained by restored plasticity. Forcing newcomers to relearn the language from the population improves it (+0.19 held-out accuracy, 95% CI [0.004, 0.409]), whereas restoring fresh-scale weights while keeping latent knowledge does not. We report two places where our pre-registered wording overclaimed or needed tightening, and how we handled them.

## The EI profile

| Criterion | Question | Verdict |
|---|---|---|
| C1 Beyond the objective | Does the system do something its objective never rewarded? | **PASS** (3 rounds) |
| C2 Ongoing emergence | Does it keep developing after convergence? | **PASS** (3 rounds); mechanism: cultural transmission |
| C3 Interaction as source | Does interaction add capability at equal compute? | **FAIL** (2 rounds) |
| C4 Self-organized structure | Does unspecified structure form above baseline? | **FAIL** (20 seeds) |
| C5 Jumps survive smooth metrics | Does an apparent capability jump hold up? | **MIRAGE** |

## How the study was done

- **Task:** a sender describes an object (4 attributes × 8 values) with a 6-symbol message; a receiver reconstructs it. 10% of attribute combinations are never seen in training.
- **Agents:** GRU networks with discrete symbols, trained alone, in populations of 4–8, and with periodic replacement of agents ("turnover").
- **Controls:** matched-size weight perturbation (v0.2) and shrink-and-perturb (v0.3), which restores learning ability while keeping latent knowledge.
- **Statistics:** seeds as the unit of replication; 95% bootstrap confidence intervals; Mann–Whitney U as a secondary test.
- **Pre-registration:** each round's design and verdict rules were committed to git before that round ran. Seeds were never reused (0–4, 100–109, 200–219).
- **Compute:** one NVIDIA RTX 5090; 256 runs in total.

## Files in this record

- `ei_bench_emergent_intelligence.pdf`: the preprint (7 pages).
- `ei-bench-results-v0-v0.3.tar.gz`: every per-run result file (256 runs), the scored reports and figures for each round, the definition with all pre-registrations, and the git history showing that the rules came before the results.

## Code

EI-Bench (Apache-2.0): https://github.com/amitb-quantum/ei-bench

## Acknowledgements

Experiment design, code and drafting were carried out with the assistance of an AI system (Claude, Anthropic). All experiments were run, checked and approved by the author, who takes responsibility for the content.
