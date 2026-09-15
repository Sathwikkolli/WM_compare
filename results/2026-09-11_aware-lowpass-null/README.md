# Why lowpass had no data in Phase B — and what it exposed

**Status:** complete. Led directly to the Phase B v2 re-run
(`results/2026-09-15_informed-detection-v2`).

## Question

Phase B (`2026-08-28_informed-detection`) produced **0/50** clips for `lowpass`.
The diagnostic showed the clean-audio (null) threshold at the first strength was
**0.9948** for blind and **0.4969** for informed. How can unwatermarked audio score
that high?

## Scripts

| script | what it measures |
|---|---|
| `informed/lowpass_probe.py` (+ `.sbatch`) | AWARE conf on 50 **unwatermarked** Phase B clips, lowpass vs highpass at 10 cutoffs |
| `informed/null_probe.py` (+ `.sbatch`) | both detectors on 200 clean (Phase B null set) + 50 watermarked clips, 5 scores, same cutoffs |

Outputs on Great Lakes: `probe.csv`, `summary.md`, `null_probe.csv`,
`null_probe_summary.md`, `audio/` (listening samples).

## Findings

### 1. The lowpass axis ran backwards (a bug)

`strength_axis.py` gave lowpass `lo=0.02, hi=0.50`, copied from highpass. For a
lowpass the cutoff is where content is deleted *above*, so 0.02 (= keep < 441 Hz)
is the **strongest** setting. Phase B's "weakest" test was a low hum.

### 2. AWARE gives false alarms on the hum (a real AWARE weakness)

Unwatermarked speech, 50 clips:

| lowpass keeps | conf median | clips > 0.5 | bit acc |
|---|---|---|---|
| everything → 1.5 kHz | 0.05–0.08 | **0/50** | 0.50 |
| < 1.1 kHz | 0.21 | 3/50 | 0.60 |
| < 662 Hz | 0.46 | 23/50 | 0.40 |
| **< 441 Hz** | **0.87** | **48/50** | **0.40** |

Highpass at the same cutoffs never exceeded 0.30.

**Why it is possible:** AWARE's confidence is
`sigmoid(mean |bit activations|)` (`aware/service/detect.py`, `full_length`
mode; `x0 = 0.041`, `k_R = 91.55`). It measures how **strongly** the network
answers, never whether the bits match the key — note 0.87 conf with bit accuracy
at chance. The curve is steep: raw 0.034 → 0.27, raw 0.091 → 0.99.

Matches `2026-08-14_detector-null-test` (440 Hz tone → 0.9679). What the
trigger is exactly (tonal content vs loud energy below the 500–4000 Hz band) is
**not** separated by this run; the spectral-flatness column was uninformative.

### 3. AWARE's genuine lowpass limit is ~1.5–2.2 kHz

Watermarked clips (200-clip null, 1% FPR): detected 100% down to 3.3 kHz, **86%**
at 2.2 kHz, **28%** at 1.5 kHz, ~0% below 1.1 kHz.

### 4. The informed detector had two measurement errors

Phase B's clean-audio informed threshold was ~0.5–0.6 at **every** lowpass
strength (vs ~0.009 for gaussian noise), reaching 0.94 on clean audio.

| cause | evidence |
|---|---|
| **Reference contains voice.** AWARE embeds at 16 kHz; `wm − org` at 22.05 kHz includes minus the voice above 8 kHz | share of reference energy above 8 kHz: median 1.2%, **max 62.3%**; the 1% threshold is the 2nd-highest of 200, so outlier clips set it |
| **Windowing.** Windows gated on reference energy, not residual energy; near-silent residual windows vote fully | whole-clip scoring lowers the threshold at mid cutoffs |

Neither fix alone suffices. Both together (`informed_detector.score_16k`):

| lowpass keeps | blind | informed (Phase B) | informed (16 kHz, whole clip) |
|---|---|---|---|
| 9.9 kHz (≈ no attack) | 100% | **78%** (thr 0.474) | **100%** (thr 0.048) |
| < 3.3 kHz | 100% | 56% | **100%** |
| < 2.2 kHz | 86% | 50% | 84% |
| < 1.5 kHz | 28% | 32% | **58%** |
| < 1.1 kHz | 4% | 2% | 4% |

### 5. A real limit of subtraction, not a bug

Below ~1.5 kHz clean informed scores stay at 0.35–0.5 even after the fix.
Lowpass and highpass at the same cutoff give **equal and opposite** clean medians
(+0.465 / −0.483 at 1.1 kHz): the statistic tracks the voice, because AWARE
shapes its watermark to the voice (its paper: a "level-proportional perceptual
budget"). Consistent with Barni et al. (2001) on non-additive watermarks.

### 6. Highpass is non-monotone for blind detection

| highpass keeps | blind detected | bit acc |
|---|---|---|
| > 441 Hz … > 2.2 kHz | 100% | 1.00 |
| > 4.4 kHz | **2%** | 0.60 |
| > 9.9 kHz (−81 dB) | **100%** | 1.00 |

Filter leakage leaves a faint watermark copy that AWARE (level-invariant) reads
perfectly. Phase B's bisection assumed monotone detection, so its highpass
crossings are unreliable. Informed loses on highpass under every scoring variant.

## Consequences

- Phase B `lowpass` "no data" = findings 1 + 2.
- Phase B "informed loses on filters" is wrong for lowpass (finding 4), right for
  highpass (findings 5–6). Codec losses (mp3/opus) are suspect for the same reason.
- Additive-noise gains (~20 dB) are not affected by findings 4–5.
- Nine pipeline errors fixed and Phase B re-run: see
  `results/2026-09-15_informed-detection-v2/README.md`.
