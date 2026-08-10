# AWARE detection A/B — watermarked vs. unwatermarked

**Status:** planned — not yet run
**Date:** 2026-08-10
**Code:** `WM_compare/ab_aware/`

## Question

When AWARE says "watermarked", is it detecting the watermark — or is it
responding to speech in general?

Every robustness benchmark in this repo so far (`bench_aware.csv`,
`extra_aware.csv`, `babar_aware_results.csv`, `real_attacks_summary.csv`) scores
**only watermarked audio**. That measures how often the detector fires, never
how often it fires when it should not. A detector that returned "watermarked"
unconditionally would score 100% on every one of those tables and nothing in
them would reveal it.

This run adds the missing half: 20 clips that were never watermarked.

## Why it matters

Confidence without a negative control is uninterpretable. Until unwatermarked
audio is scored under the same conditions, no detection number in this project
has a false-positive rate attached, and the operating threshold (`conf >= 0.5`
in `detect_aware.py`) has never been checked against anything.

## Design

**Two arms, 40 Emilia clips, disjoint (NOT matched pairs)**

| Arm | n | What it is |
|---|---|---|
| `wm` | 20 | AWARE-embedded, per-clip random 20-bit payload |
| `clean` | 20 | untouched — the negative control |

Disjoint arms were the design call. The consequence, recorded up front: clip
identity is confounded with arm, so between-arm differences carry clip-level
variance as well as the watermark, and **no paired test is valid across arms**.
Both arms are drawn from one shuffled pool under the same ≥9 s filter, which is
what keeps them distributionally comparable. The McNemar tests in the analysis
are *within* the `wm` arm, where the pairing is real.

Per-clip random payloads rather than one shared payload: a shared pattern would
let a detector biased toward that pattern post inflated bit accuracy across the
whole arm, invisibly.

**Conditions (11) — one strength each, chosen to span failure families**

| Family | Conditions |
|---|---|
| control | `clean` |
| additive | `gaussian_noise` 20 dB |
| codec | `mp3` 32k, `opus` 16k, `aac` 40k |
| band-limit | `lowpass` 0.3, `highpass` 0.2 |
| channel | `echo` 0.3 |
| requantize | `quantization` 8 lvl |
| desync | `time_stretch` 1.1, `crop_50` |

One strength per attack, not a sweep: 40 clips cannot support a strength sweep
and a detection A/B at once. `crop_50` is proportional rather than the absolute
`crop_30s` used in `run_bench.py`, because Emilia clips run ~10 s and a 30 s crop
is undefined here.

**Grid:** 40 clips × 11 conditions = 440 decisions. Every clip is scored under
every condition — nothing is randomly subsampled, so the confusion matrix rests
on the full grid rather than a handful of draws.

## Metrics

| Metric | What it tells you |
|---|---|
| Confusion matrix @ 0.5 | The codebase's current default. Arbitrary, reported for continuity. |
| Confusion matrix @ calibrated | Threshold derived from the negatives (zero observed FP). **The operating point you would actually ship.** |
| AUC + clip bootstrap CI | Threshold-free separation. Immune to the threshold argument. |
| Clip-level FPR + Wilson CI | **The only false-positive number here with an honest interval.** |
| EER | Where FPR meets FNR. Coarse at this n. |
| Bit accuracy vs. binomial null | 50% is chance on 20 bits. Raw accuracy alone is not evidence. |
| McNemar + Holm | Does each distortion *significantly* hurt, correcting for 10 tests. |
| PESQ / SNR | What the watermark costs perceptually. Robustness without this is not interpretable. |

Deliberately **not** reported: TPR @ FPR=1e-3. With 20 negatives the finest
resolvable FPR is 1/20 = 5%; anything below that would be invented.

## Figures

1. ROC, pooled — the headline single figure
2. Confidence distribution by arm, with both thresholds marked — explains every other number
3. Per-condition TPR and AUC bars
4. Bit accuracy against the 50% chance floor

## Prediction (recorded before running, so it's falsifiable)

1. **Clean-condition separation is near-total.** AUC > 0.98 on the `clean` row,
   zero false positives on the clean arm. If this fails, something is wrong with
   the embed step, not with the science.
2. **`time_stretch_1.1` is the worst cell**, at or near chance (AUC ~0.5, bit
   accuracy ~0.5, not significant against the binomial null). `align_bench`
   established that no aligner recovers time-stretched audio; AWARE has no
   reason to do better blind.
3. **`crop_50` degrades but survives** — meaningfully above chance. It is
   recoverable in principle (`gcc_phat` was sample-exact on crop), so a low score
   here means *misaligned*, not *destroyed*. Those are different failures and
   should not be reported as one.
4. **The calibrated threshold lands below 0.5**, meaning the default costs
   detections it did not need to. If it lands above 0.5, the default is
   generating false positives.
5. **Codecs at these rates cost some bit accuracy but little detection** —
   confidence degrades more gracefully than payload.

## Limits (known before running, not excuses added after)

1. **20 negatives cannot certify a low FPR.** A perfect 0/20 still leaves a 95%
   Wilson interval of [0, 16%]. This run can catch a *gross* false-positive
   problem; it cannot support "FPR < 1e-3". Fixing that means scaling the clean
   arm to 200+, which is cheap — negatives skip the embedding step entirely.
2. **440 rows are ~40 independent units.** Eleven conditions share one clip, so
   every interval resamples clips, and the pooled confusion counts are
   descriptive only.
3. **~10 s clips only.** Same regime limit as `align_bench`; the 3-minute case
   is uncovered.
4. **One strength per attack.** Where a cell fails, this run does not say how
   far it was from passing.

## Reproduce

```bash
conda activate wmcompare && cd $WM_COMPARE_BASE/ab_aware
python make_clips.py && python embed.py    # login node
sbatch ab_aware.sbatch                     # array 0-39
python analyze.py                          # after the array finishes
```
