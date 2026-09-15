# Phase B v2 — re-run after fixing measurement errors

**Status:** submitted, awaiting results. Nothing below the "Results" heading is
filled in until the run completes.

The registered first run lives in `results/2026-08-28_informed-detection` and is
**not overwritten**. This directory is a re-run with the errors below fixed.
Both are reported.

**Scope: `lowpass` only.** All nine fixes are in the code, but only lowpass is
re-run here — it is the attack the investigation was about. The other 21
attacks keep their run-1 results, which are still affected by errors 2–7 (and
`volume_down`/`volume_up` remain unmeasured). A full re-run is
`bash submit_phase_b.sh --from nullcal` with no `--attacks`.

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
bash submit_phase_b.sh --from nullcal --attacks lowpass   # null cache in real_audio/ is reused
```

Outputs:

- `summary_phase_b.md`, `figures/*.png` — blind vs registered informed
- `summary_phase_b_informed16.md`, `figures/*_informed16.png` — blind vs corrected informed
- `data/null/null_lowpass.csv`, `data/null/nullraw_lowpass.csv`, `data/sweep/sweep_clip*.csv`

## What to check first

1. **No `no data`** for lowpass in section 0 of either summary.
2. Blind crosses around cutoff 0.07–0.10 (~1.5–2.2 kHz), as the null probe found.
3. Registered informed vs `informed16`: the probe predicts `informed16` outlasts
   blind (58% vs 28% detected at 1.5 kHz) while the registered arm does not.
4. Few or no `NON_MONOTONE` rows — lowpass was monotone in the probe.

## Results

*(empty until the run completes)*
