# Experiment Index

Every experiment run in this project, newest first. One row per directory in
`results/`. See `README.md` for the archival convention.

| Date | Experiment | Headline finding | Status |
|---|---|---|---|
| 2026-08-10 | [aware-detection-ab](2026-08-10_aware-detection-ab/) | First **negative control** for AWARE — 20 watermarked vs. 20 unwatermarked clips across 11 conditions. Answers whether the detector fires on the watermark or on speech. | **planned** |
| 2026-08-05 | [align-method-bakeoff](2026-08-05_align-method-bakeoff/) | **`gcc_phat` wins** — sample-exact on crop, 0% false-shift, best confidence, 20x faster than audalign. `dtw_subseq` disqualified (22.8% false-shift). **No method aligns time-stretched audio.** See [ANALYSIS.md](2026-08-05_align-method-bakeoff/ANALYSIS.md) | **complete** |

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
