# Changes

## v0.3 (2026-10-30)
- DEFINITION.md §9: v0.3 pre-registered (C2c shrink-and-perturb control, C2-rep2, C4 with 20
  seeds), including a calibration of the control run before any v0.3 experiment.
- `turnover_mode="shrink_perturb"` (`sp_sigma`, default 0.1) and `shrink_perturb_agent()`;
  per-event diagnostics (norm ratio, cosine to old weights) in each run's turnover events.
- Suite `v0.3` (now the default) -> `results/v03`: c2c (3 arms x 10 seeds, 24k steps) and c1 (20 seeds).
- `scripts/worker.sh`: workers restart themselves after a crash (up to 5 times); `launch.sh` uses it.
- Report: v0.3 sections, pre-registered C2c reading, shrink-and-perturb diagnostics.
- v0.2 run note: one worker died early from a transient CUDA "unknown error"; its run was
  reclaimed and completed by another worker (160/160 complete). Motivated the auto-restart.
- Note: the new `sp_sigma` config field changes run-id hashes, so re-running an earlier suite
  will not reuse its old JSONs. Existing results are unaffected and still load in the report.

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
