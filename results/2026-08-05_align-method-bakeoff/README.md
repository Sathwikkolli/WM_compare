# Alignment method bake-off

**Status:** planned — not yet run
**Date:** 2026-08-05
**Code:** `WM_compare/align_bench/`

## Question

Which audio alignment method should we use to line up the unwatermarked
original against a distorted watermarked copy, so that non-blind (informed)
watermark detection becomes possible?

## Why it matters

We hold the *unwatermarked* original. Every benchmark in the literature
(AudioMarkBench, SoK 2025, omnisealbench) evaluates these watermarks **blind** —
detector sees only the suspect audio. Non-blind detection is strictly more
powerful because subtracting an aligned original removes *host interference*,
which is the dominant noise term for a blind detector. Alignment is the gate:
misalign by a few samples and subtraction doubles the signal instead of
cancelling it.

**Validity constraint** (agreed up front, so the result isn't circular): the
original may only be used to estimate a small set of generic inverse parameters
— time offset, speed factor, gain, EQ curve. Never to inject signal. Litmus
test: *could this same correction be applied if an oracle just told us the
attack parameters, without the audio?* If no, it's cheating.

## Design

**Methods (6)**

1. GCC-PHAT (scipy, ~15 lines) — control/floor, sample-accurate, single offset
2. audio-offset-finder (BBC) — MFCC cross-correlation
3. audalign · FingerprintRecognizer — the only multi-segment-capable method
4. audalign · CorrelationRecognizer
5. audalign · CorrelationSpectrogramRecognizer
6. librosa subsequence DTW — warping cases

Excluded: Panako, Olaf, NeuralFP, GraFP, Chromaprint. All solve
retrieval-from-a-database; we align against one known reference.

**Attacks (20 types, 77 strength configurations)**

Metapyxl mastering proxy, fixed: `dynamic_compression` (2), `echo` (5),
`mp3` (5), `quantization` (5), `lowpass` (5), `gaussian_noise` (5)

Desync, added by hand (VoxWatermark has **no cropping attack**):
`crop_head` (4), `crop_tail` (4), `splice_cut` (3), `insert_silence` (3),
`insert_foreign` (3)

Desync, from VoxWatermark: `time_stretch` (5), `time_jitter` (2),
`phase_shift` (2)

Codec / zero-shift stress: `opus` (6), `aac` (2), `encodec` (5),
`background_noise` (5), `highpass` (5), `inverse_polarity` (1)

`inverse_polarity` is the trap: true offset 0, but flipping the sign makes a
naive `argmax` of cross-correlation fail outright.

**Clips (30)** — Emilia only, natural durations, no concatenation.
Manifest: 896,592 rows -> 435,788 after the 9 s filter -> 30 sampled (seed 0),
plus 5 held-out clips as foreign audio for `insert_foreign`.

Actual durations: **min 9.2 s, median 14.5 s, max 27.8 s**. Wider than the ~10 s
we assumed, so there is some length variation to read a scaling trend from --
but nothing over 60 s, so the 3-minute regime is still uncovered.

**Run matrix**

```
Phase 1 (screen)   5 clips x 77 configs x 6 methods = 2,310 runs
Phase 2 (full)    30 clips x 77 configs x ~3 methods = 6,930 runs
```

## Metrics

| Metric | What it tells you |
|---|---|
| Offset error (median, p95) | Raw accuracy. p95 catches catastrophic failures the median hides. |
| Hit rate @ 20 ms / @ 1 ms | Practical pass rate. **Headline number.** |
| False-shift rate | Does it hallucinate offsets when nothing moved? **>5% disqualifies.** |
| Confidence AUC | Can we trust it to know when it failed? Enables fallback. |
| Segment F1 | Multi-segment capability (splice cases only). |
| Runtime + scaling | Usability at 120 s. |

Decision order: false-shift gate -> hit rate -> confidence AUC -> segment F1
-> runtime veto.

## Figures

1. Error CDF (tolerance vs fraction within it) — the single best figure
2. Heatmap: method x attack, hit rate @ 20 ms
3. Robustness-vs-strength curves, one panel per attack
4. Predicted vs true offset scatter — off-diagonal clusters reveal *why* it failed
5. Reliability plot (confidence vs observed accuracy)
6. Alignment ribbon, ~10 hand-picked splice cases, qualitative

## Prediction (recorded before running, so it's falsifiable)

GCC-PHAT wins the pure-shift and zero-shift families outright. audalign
fingerprinting is the only method scoring above zero on multi-segment.
Time-stretch and jitter need DTW or fail everywhere. Expected outcome is a
**two-method combination**, not a single winner.

## Tolerances

Three, because one number would mislead:

| Bar | Why |
|---|---|
| 1 ms | strict; only matters if something needs sample-grade sync |
| 20 ms | matches `W_MS` in `fsss/exp_a_repeatability.py`, for comparability |
| **50 ms** | **headline.** AWARE(20bps) slides its detector in `chunk_duration//24` = 42 ms steps, so finer alignment buys the detector nothing. Also the conventional bar in the music-sync literature. |

This matters because `methods.calibrate()` measured audalign's fingerprint and
spectrogram recognizers at a **22–25 ms resolution floor on clean audio**.
Scoring only at 20 ms would mark them near-zero for being coarse rather than
wrong.

Measured residuals on a clean synthetic 1 s crop:

| method | residual |
|---|---|
| `gcc_phat` | 0.0 ms |
| `aof` | 0.0 ms |
| `audalign_corr` | 0.0 ms |
| `dtw_subseq` | 8.0 ms |
| `audalign_spec` | 21.7 ms |
| `audalign_fp` | 24.8 ms |

## Setup resolved

- Emilia manifest: `/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv`
  (columns: `path, duration_s, speaker, language, dnsmos, dataset`)
- Env: `wmcompare`, with `audalign` + `audio-offset-finder` installed on top.
  This upgraded librosa 0.9.2 -> 0.11.0, breaking AWARE's pin; AWARE still loads
  and runs, but that pin is now violated -- re-verify before trusting new
  AWARE numbers.
- Smoke test (clip 0, 3 attacks, 8 configs, 6 methods): 48 rows, 0 skipped, 12.2 s.

## Known coverage gap

Emilia is speech only. The repetitive-music failure mode — where several
correlation peaks tie and alignment picks the wrong one — goes untested. The
Metapyxl client chain includes a music bed, so this may matter in production.
