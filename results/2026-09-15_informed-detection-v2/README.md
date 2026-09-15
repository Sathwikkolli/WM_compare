# Phase B v2 — re-run after fixing measurement errors

**Status:** submitted, awaiting results. Nothing below the "Results" heading is
filled in until the run completes.

The registered first run lives in `results/2026-08-28_informed-detection` and is
**not overwritten**. This directory is a re-run of the same experiment with the
errors below fixed. Both are reported.

## Why a re-run

`results/2026-09-11_aware-lowpass-null` investigated why `lowpass` produced no
data. It found two errors, and reading the pipeline afterwards found seven more.

| # | error | effect on the first run | fix |
|---|---|---|---|
| 1 | `lowpass` axis ran backwards (t=0 was cutoff 0.02 = "keep < 441 Hz") | lowpass: 0/50, every clip failed at "t=0" | `lo=0.45, hi=0.02` in `strength_axis.py` |
| 2 | Informed reference `wm − org` at 22.05 kHz contains the voice above 8 kHz (median 1.2%, max 62% of its energy); windowed mean over-weights near-silent residual windows | clean-audio informed threshold ~0.5 on filters (0.009 on noise); at lowpass 0.45 only 78% of watermarked clips detected | new arm `informed16`: whole-clip correlation at 16 kHz (`informed_detector.score_16k`). Clean threshold 0.474 → 0.048, detection 78% → 100% |
| 3 | `null_calibrate` looped over `attacks_screen.SCREEN_GRID` names, which say `volume`, not `volume_down`/`volume_up` | those 2 attacks never calibrated and silently dropped: 20 attacks ran, not 22 | loop over `strength_axis.AXIS` |
| 4 | Bisection checked only the two ends, assuming detection only gets worse | `highpass` (blind fails at 0.15–0.3 and recovers at 0.45) got a fabricated crossing or a false "survived" | 4 interior probes; a recovery is reported as `NON_MONOTONE`, never as a crossing |
| 5 | "Blind won" counted only paired clips | blind wins undercounted (`smooth`, `dynamic_compression`, `echo`, …) | `score_phase_b.outcome()` decides every status pair |
| 6 | "One detector failed at the weakest setting, the other worked" shown as no data | wins hidden for both detectors | same `outcome()` |
| 7 | Overview bar = mean on 0–1 scale, label = mean in native units | `resample_roundtrip` bar and label disagreed in sign | medians for both |
| 8 | `null_calibrate` never passed `raw_writer` | `nullraw_*.csv` never written, `null_separation.png` impossible | written for every attack |
| 9 | A re-run would overwrite the registered result | original lost | `PHASEB_RUN` env var; this directory |

## What is registered and what is not

- **Blind** and **informed** (windowed, 22 kHz) are exactly the registered arms.
  Fixes 1, 3–9 change how they are *measured and counted*, not what they are.
- **informed16 is post-hoc.** It was chosen after seeing that the registered
  statistic mis-measures filters. Part of it is a bug fix (the reference should
  be the watermark), part is a design change (whole clip vs windows). It is
  labelled as a correction everywhere it appears.

## Run

```
cd ~/wm_compare/informed
bash submit_phase_b.sh --from nullcal     # null cache in real_audio/ is reused
```

Outputs:

- `summary_phase_b.md`, `figures/*.png` — blind vs registered informed
- `summary_phase_b_informed16.md`, `figures/*_informed16.png` — blind vs corrected informed
- `data/null/null_*.csv`, `data/null/nullraw_*.csv`, `data/sweep/sweep_clip*.csv`

## What to check first

1. **No `no data`** in section 0 of either summary. Any is a pipeline failure.
2. `volume_down` and `volume_up` appear (22 attacks, not 20).
3. `lowpass` has crossings for blind (expected ~1.5–2.2 kHz, i.e. cutoff 0.07–0.10).
4. `highpass` shows `NON_MONOTONE` for blind on most clips.
5. Additive attacks: does the ~20 dB informed gain from run 1 hold under both informed arms?
6. `mp3`, `opus`, `aac`: does informed16 remove the informed loss seen in run 1?

## Results

*(empty until the run completes)*
