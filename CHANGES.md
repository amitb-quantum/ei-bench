# Changes

## v0.2 (2026-10-29)
- DEFINITION.md: v0 results recorded (§6); v0.2 tests pre-registered (§7) before any v0.2 run.
- New experiments (suite `v0.2`, default): c2b (reset vs. displacement-matched perturbation vs. none),
  c3b (heterogeneous learning-rate populations, N=4 and N=8, plus homogeneous replicate),
  c5b (12-point hidden-size sweep, transition-width test), c1 replication; fresh seeds 100–109.
- `turnover_mode="perturb"` and `lr_spread` config options; `perturb_agent()`.
- Runner: lock-file work queue so several workers share one grid; `--suite`, `--progress`;
  `scripts/launch.sh N` starts N workers.
- Report: v0.2 sections, Mann–Whitney U as a secondary statistic, transition-width bootstrap.
- Note: new config fields change run-id hashes, so re-running `--suite v0` would not reuse the
  existing v0 JSONs. v0 results in `results/full` are unaffected and still load in the report.

## v0.1 (2026-10-29)
- Initial release: C1–C5 harness, v0 grid.
