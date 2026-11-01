#!/usr/bin/env bash
# Build the dataset bundle for the Zenodo dataset record: every per-run JSON, report,
# summary and figure from v0, v0.2 and v0.3, plus the definition and its full git history.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="ei-bench-results-v0-v0.3"
rm -rf "$OUT" "$OUT.tar.gz"
mkdir -p "$OUT"
for d in full v02 v03; do
  [ -d "results/$d" ] || { echo "missing results/$d"; exit 1; }
  mkdir -p "$OUT/results/$d"
  cp results/$d/*.json results/$d/REPORT.md results/$d/*.png "$OUT/results/$d/"
done
cp DEFINITION.md README.md LICENSE NOTICE CITATION.cff "$OUT/" 2>/dev/null || true
git log --date=iso --format='%H %ad %s' > "$OUT/GIT_HISTORY.txt"
git log --date=iso --format='%H %ad %s' -- DEFINITION.md > "$OUT/PREREGISTRATION_HISTORY.txt"
n=$(ls "$OUT"/results/*/*.json | grep -vc summary.json)
cat > "$OUT/README_DATASET.md" <<TXT
# EI-Bench v0–v0.3 results

Per-run results for the preprint "Emergent Intelligence, Defined So It Can Fail" (DOI 10.5281/zenodo.23849135).

- results/full : v0 (46 runs, seeds 0-4)
- results/v02  : v0.2 (160 runs, seeds 100-109)
- results/v03  : v0.3 (50 runs, seeds 200-219)

Each <run_id>.json holds the full config, the untrained baseline, the training history,
turnover events and final metrics. REPORT.md and summary.json in each folder are the scored
verdicts. DEFINITION.md holds the pre-registrations; PREREGISTRATION_HISTORY.txt lists the
git commits that changed it, with timestamps, so the order "rules first, results second"
can be checked. Code: https://github.com/amitb-quantum/ei-bench (Apache-2.0).

Run files in this bundle: $n
TXT
tar czf "$OUT.tar.gz" "$OUT"
echo "wrote $OUT.tar.gz with $n run files (expect 256)"
sha256sum "$OUT.tar.gz"
