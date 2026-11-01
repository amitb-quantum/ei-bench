# Zenodo record for EI-Bench

**One record**, of type Publication → Preprint, with two files: the paper PDF and the results
bundle. A software DOI can be added later through the Zenodo–GitHub integration, once the repo is
public.

Publishing on Zenodo is permanent: a published version can't be deleted, only superseded by a new
version. Keep the record as a draft until the repo is public, so every link resolves on day one.

## Record (reserved DOI 10.5281/zenodo.23849135)

| Field | Value |
|---|---|
| Resource type | Publication → Preprint |
| Title | Emergent Intelligence, Defined So It Can Fail: A Pre-registered Test in AI Agents That Invent Their Own Language |
| Publication date | 2026-10-31 |
| Creators | Brahmbhatt, Amit · ORCID 0009-0003-5882-8476 · Affiliation: Quantum Clarity LLC |
| Version | 1.0 |
| Language | English |
| License | Creative Commons Attribution 4.0 International (CC BY 4.0) |
| Description | Paste `paper/zenodo_description.md`. If the editor shows raw `##` and `**` symbols, use its source-code (`<>`) button and paste `paper/zenodo_description.html` instead |
| Files | `paper/ei_bench_emergent_intelligence.pdf` and `ei-bench-results-v0-v0.3.tar.gz` |
| Related works | *Is supplemented by* → URL https://github.com/amitb-quantum/ei-bench (resource type: Software) |

**Keywords:** emergent intelligence · emergent communication · pre-registration · iterated learning ·
cultural transmission · multi-agent systems · compositionality · emergent abilities · metric mirage ·
shrink-and-perturb · neural plasticity · reproducibility · EI-Bench

Build the results bundle:
```bash
cd ~/projects/ei-bench && bash scripts/make_zenodo_bundle.sh   # expect "256 run files"
```

## Later: software DOI via GitHub (optional)

1. Make the repo public:
   `gh repo edit amitb-quantum/ei-bench --visibility public --accept-visibility-change-consequences`
2. Zenodo → GitHub (logged in as amitb-quantum) → switch ON `amitb-quantum/ei-bench`.
3. `gh release create v0.3.0 --title "EI-Bench v0.3.0" --notes "Code for the preprint (DOI 10.5281/zenodo.23849135)."`
4. Zenodo mints a software DOI from `CITATION.cff`. Add it to the preprint as *Is supplemented by*,
   and add the preprint DOI to the software record as *Is supplement to*.
