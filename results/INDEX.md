# Experiment Index

Every experiment run in this project, newest first. One row per directory in
`results/`. See `README.md` for the archival convention.

| Date | Experiment | Headline finding | Status |
|---|---|---|---|
| 2026-08-05 | [align-method-bakeoff](2026-08-05_align-method-bakeoff/) | Which alignment method to use for non-blind watermark detection — 6 methods x 20 attacks x 30 Emilia clips | **planned** |

## Status values

- `planned` — designed, not yet run
- `running` — job submitted
- `complete` — numbers in, summary written
- `superseded` — a later run replaces it (link the replacement)
- `abandoned` — stopped; say why in the run's README

## Backfill needed

Results produced before this archive existed are scattered in the repo root and
are currently gitignored. Worth recovering into dated directories if the numbers
are still cited anywhere:

- `bench_audioseal.csv`, `bench_aware.csv`, `bench_timbre.csv` — solo robustness benchmark
- `extra_*.csv` — extended attack set
- `babar_audioseal_results.csv`, `babar_aware_results.csv` — includes `edit_trim_head5s`, `edit_splice_cut5s`
- `real_attacks_summary.csv` — real-world attack experiment
- `cascade_out/*.csv` — cascade / stacking study
- `vox_out/vox_report.html` — VoxWatermark strength sweep
- `fsss_out/exp_v12_metapxyl_compare.csv` — Metapyxl mastering-chain comparison
- `client_detection_summary.csv` — client pipeline stages
