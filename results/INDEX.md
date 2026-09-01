# Experiment Index

Every experiment run in this project, newest first. One row per directory in
`results/`. See `README.md` for the archival convention.

**The production detection threshold and its evidence live in
[THRESHOLD_DECISION.md](THRESHOLD_DECISION.md).** Read that before changing any
threshold — it is where the null test and the A/B are reconciled, and the two
disagree in a way neither run reveals alone.

| Date | Experiment | Headline finding | Status |
|---|---|---|---|
| 2026-08-28 | [informed-detection](2026-08-28_informed-detection/) | **Phase A complete.** 27 attacks, 118 configs, 50 clips. **Only 3 attacks beat detection on usable audio — all codec** (`encodec` 6kbps, `mp3` 8k, `resample_roundtrip` 4kHz), and all 3 have bit accuracy **below 0.85**, so the watermark is destroyed rather than masked. **Every additive attack was shrugged off** — music, gaussian, babble, factory, machinegun all at conf **1.00**, bit acc **1.000**. The music-bed lead candidate is dead (window −14.16 dB, 0/19 clips). **Predictions 1 and 2 refuted**; my category prediction was backwards — compression breaks AWARE, adding sound does not. **Caveat:** verdicts depended on a quality floor that listening contradicted, and 54% of music-sweep rows lost DNSMOS to a ValueError. **[Phase B](2026-08-28_informed-detection/PHASE_B_PLAN.md) drops the quality threshold entirely** — paired blind-vs-informed crossings by bisection at matched 1% FPR. See [README.md](2026-08-28_informed-detection/README.md) | **phase A complete; phase B planned** |
| 2026-08-18 | [frame-align-null](2026-08-18_frame-align-null/) | **Alignment needs ~500 ms — one AWARE hop (50 ms) is unalignable at 5.1%.** **A calibration without a null would have shipped a 30–41% false-accept rate at 250 ms** (90.6% at 50 ms) — the null is mandatory at 250–500 ms, redundant above 1000 ms. Neither method has a reject option: `accept-all` is **100%** on all 7,200 unrelated pairs. **The bake-off winner loses at frame level** — `aof` beats `gcc_phat` at every length ≥250 ms (80.3% vs 53.8% useful-accept at 250 ms) — **but only at 20 ms tolerance**: `gcc_phat` is sample-exact whenever it works, `aof` is quantised to ~7.7 ms and caps at 26% @1ms. **Predictions 3, 4 and 5 refuted.** Phase 1 (clean, native refs) only. See [README.md](2026-08-18_frame-align-null/README.md) | **complete** (phase 1; phase 2 pending) |
| 2026-08-17 | [attack-damage-control](2026-08-17_attack-damage-control/) | **The A/B's two failures are destroyed audio, not a fragile watermark.** Unwatermarked audio is damaged identically — largest arm gap over 15 conditions is **0.06 PESQ**. `highpass_0.2` leaves unwatermarked speech at PESQ **1.32**, `quantize_8lvl` at **1.04** (clean 4.64). **And AWARE outlives usability:** audio crosses PESQ 2.0 at ~850 Hz / ~144 levels, AWARE only loses it at **2500 Hz / 8 levels** — at `hp_1000hz` (PESQ 1.81) it is still 5/5 at conf 1.000 with perfect bits. Mechanisms are opposite: high-pass empties the 1000–4000 Hz band (1.9% left), quantisation floods it (251%). See [summary.md](2026-08-17_attack-damage-control/summary.md) | **complete** |
| 2026-08-14 | [detector-null-test](2026-08-14_detector-null-test/) | **Keep the 0.5 threshold.** 0 false positives in 300 clean Emilia clips; no speech file anywhere exceeded **0.2723** (headroom +0.23, AudioSeal +0.35). Refutes the "clean speech scores 0.4–0.55" claim, so 0.65 is unnecessary and would cost sensitivity. **One defect:** a pure 440 Hz tone scores AWARE **0.9679** — tonal input, no threshold fixes it. See [summary.md](2026-08-14_detector-null-test/summary.md) | **complete** |
| 2026-08-10 | [aware-detection-ab](2026-08-10_aware-detection-ab/) | **AUC 0.9747** [0.9605, 0.9871], 0 false positives, **TPR 80.9% @ 0.5** over 11 conditions. 8 of 11 perfect; `highpass_0.2` (0/20) and `quantize_8lvl` (1/20) fail significantly (Holm p=0.0003), both at PESQ ≈ 1.0–1.3. **Its calibrated 0.2463 is refuted** by the null test's 300 negatives reaching 0.2723 — see [THRESHOLD_DECISION.md](THRESHOLD_DECISION.md) | **complete** |
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
