# Attack-damage control — is the audio destroyed, or is AWARE fragile?

**Status:** complete — see [summary.md](summary.md)
**Date:** 2026-08-17
**Code:** `WM_compare/damage_ab/`

> **Result in one line:** both failing conditions destroy the audio equally in
> both arms (arm gaps ≤ 0.06 PESQ everywhere), so neither is evidence about
> AWARE — and AWARE holds well past the point the audio stops being usable.

## Question

[2026-08-10_aware-detection-ab](../2026-08-10_aware-detection-ab/) found two
conditions where AWARE fails: `highpass_0.2` (0/20) and `quantize_8lvl` (1/20).
It scored **only watermarked audio, at one strength each**. Two explanations fit
that result equally well and it cannot separate them:

1. **AWARE is fragile** to band-limiting and requantisation, or
2. **the attacks destroy the audio**, and nothing could survive in a file nobody
   would listen to.

The distinction decides whether these are defects to fix or attack settings to
document. This run answers it by putting **unwatermarked audio through the
identical attacks** and sweeping the strength.

## Why the existing evidence is not enough

`THRESHOLD_DECISION.md` already notes both failures sit at PESQ ≈ 1.0–1.3 against
4.29 clean, and calls that "a mitigating fact, not an excuse". But that PESQ is
`src → attack(wm)`: it mixes watermark cost and attack damage into one number.
Nothing measured so far scores an **unwatermarked** file under these attacks, so
the claim "the audio was already destroyed" is currently an inference, not a
measurement.

## Design

**Matched pairs, 5 Emilia clips (≥9 s), 15 conditions = 75 rows**

| Arm | What it is |
|---|---|
| `src` | the clip, unwatermarked — **the control** |
| `wm` | the same clip, AWARE-embedded, per-clip random 20-bit payload |

Matched, not disjoint — the opposite of the A/B's call, for a different question.
This is a within-clip quality comparison, so pairing removes clip variance that
would otherwise swamp the effect at n=5. **The consequence, recorded up front:
this run cannot produce a false-positive rate.** Its `src` arm is a paired
control, not an independent negative sample. FPR lives in
[2026-08-14_detector-null-test](../2026-08-14_detector-null-test/) (n=300).

**Conditions**

| Group | Values | Why |
|---|---|---|
| control | `clean` | watermark cost with no attack |
| anchor | `mp3_32k` | a condition AWARE survives 20/20 in the A/B — fixes the PESQ scale on these clips |
| high-pass | 200, 500, 1000, 1500, 2000, 2500, **3200** Hz | 200/500 sit below AWARE's band; 1000+ eat into it. **3200 Hz reproduces the A/B's `highpass_0.2`** |
| quantize | 256, 64, 32, 16, **8**, 4 levels | **8 reproduces the A/B's `quantize_8lvl`**; 256 is near the bench's passing `quantize_8bit` |

Cutoffs are specified in **Hz** and the ratio derived, because `a_highpass` takes
a fraction of the sample rate and that indirection already produced one error in
this repo (`THRESHOLD_DECISION.md` recorded `highpass_0.2` as a 1600 Hz cutoff; at
16 kHz it is 3200 Hz — corrected there on 2026-08-17). `sweep.py` asserts at
import that its two anchor cells
equal the A/B's, so the sweep cannot silently drift off the run it is anchored to.

## Metrics

**Four PESQ measurements per row**, each against a different reference. This is
the core of the design — one PESQ number cannot answer the question:

| Column | Reference → Degraded | Tells you |
|---|---|---|
| `wmcost_pesq` | `src` → `wm` | what the watermark costs. No attack. |
| `src_pesq` | `src` → `attack(src)` | **the control.** Attack damage with no watermark present |
| `wm_pesq` | `wm` → `attack(wm)` | attack damage to the watermarked signal — compare directly with `src_pesq` |
| `comb_pesq` | `src` → `attack(wm)` | both together. What the A/B reported. |

Plus SNR for each, AWARE confidence and bit accuracy on both arms, and
**AWARE-band (1000–4000 Hz) energy retention**, which separates "band emptied by
a filter" (≪1) from "band buried in switching noise" (>1). PESQ cannot tell those
apart and they are different failure mechanisms.

## Pre-registered criterion

Fixed here, before the run, so it cannot be tuned to the outcome. An attack is
**destructive** rather than watermark-specific when **both** hold:

- **(a)** `src_pesq < 2.0` — audio with no watermark in it is already unusable
- **(b)** `src_pesq − wm_pesq ≤ 0.3` — the watermarked arm is not materially worse

(a) alone would only say the attack is harsh; (b) is what rules out "the
watermark made the audio fragile". Both constants — the 2.0 floor and the 0.3
material gap — are fixed as of this document.

## Prediction (recorded before running)

1. **`hp_3200hz` is destructive.** Measured on `audio/aware_wm.wav` locally
   (2026-08-17): a 3200 Hz high-pass leaves **1.89%** of the 1000–4000 Hz band's
   energy and the residual sits at 0.15 dB SNR. `src_pesq` should land near 1.3
   and the arm gap near zero.
2. **`quant_8lvl` is destructive, and worse.** Same local measurement: **98.21%
   of all samples collapse onto a single level**, the grid is not zero-centred so
   silence becomes a DC offset, and the error energy exceeds the signal energy
   (SNR **−2.9 dB**). Band energy goes *up* (~138%) — noise, not signal.
3. **The high-pass breaking point is between 500 and 1500 Hz.** 500 Hz passes in
   `bench_aware.csv` at conf 0.998; 1000 Hz is the bottom edge of AWARE's band.
4. **`quant_256lvl` and `mp3_32k` survive**, with `src_pesq` well above 2.0. If
   the anchor fails, the clip set or the pipeline is wrong, not the science.
5. **Arm gaps stay under 0.3 everywhere.** If a gap exceeds it, the watermark
   genuinely does make audio more fragile under that attack — which would be a
   real finding and would falsify the framing above.

## Outcome — how the predictions scored

Written after the run, against the predictions above, unedited.

| # | Prediction | Outcome |
|---|---|---|
| 1 | `hp_3200hz` destructive, ~1.89% band retained, arm gap ≈ 0 | **held.** src PESQ 1.32, band 1.9% (wm) / 2.4% (src), arm gap 0.01 |
| 2 | `quant_8lvl` destructive, band energy *up* | **held.** src PESQ 1.04, band 251% (wm) / 225% (src), arm gap −0.00 |
| 3 | high-pass breaking point between 500 and 1500 Hz | **FAILED.** It broke at **2500 Hz** — AWARE is substantially more robust than predicted |
| 4 | `quant_256lvl` and `mp3_32k` survive | **held**, though `quant_256lvl` lands at PESQ 2.40, only just above the floor |
| 5 | arm gaps stay under 0.3 everywhere | **held.** Largest gap anywhere is 0.06, at `mp3_32k`, in the watermark's favour |

Prediction 3 failing is the substantive news, and it failed in AWARE's favour.

## What the run establishes

**1. Neither A/B failure is watermark-specific.** Unwatermarked audio is damaged
identically — the largest arm gap over all 15 conditions is 0.06 PESQ. The
pre-registered criterion returns DESTRUCTIVE for both cells.

**2. Detection outlives usability, by a wide margin.**

| | audio crosses PESQ 2.0 | AWARE crosses conf 0.5 |
|---|---|---|
| high-pass | ≈ 850 Hz (interpolated, 500→1000) | **2500 Hz** |
| quantize | ≈ 144 levels (interpolated, 256→64) | **8 levels** |

Inside that gap AWARE is not merely surviving: at `hp_1000hz` (PESQ 1.81) it is
5/5 at conf 1.000 with bit accuracy 1.000, and at `quant_32lvl` (PESQ 1.17) it is
5/5 with perfect bits.

**3. The two mechanisms are opposite.** High-pass *empties* the 1000–4000 Hz band
(1.9% retained); quantisation *floods* it with rounding noise (251%). Both end in
"not detected". Any mitigation would have to be different for each.

**4. No false alarms.** The unwatermarked arm stayed at conf 0.04–0.14 across all
15 conditions, never approaching 0.5 — consistent with the null test, though this
run cannot produce an FPR (see Limits #2).

## Known issues with this run

- **The `verdict` column reads misleadingly on three rows.** `hp_1000hz`,
  `quant_64lvl` and `quant_32lvl` are labelled DESTRUCTIVE while AWARE detects
  5/5 with perfect bit accuracy. The criterion is judging *the attack*, which is
  what it was defined to do, but the label invites the opposite reading. The
  criterion is left exactly as pre-registered; `summary.md` needs a second column
  separating "attack destroyed the audio" from "AWARE lost the watermark". The
  gap between those two is the finding, so it should not be collapsed into one
  word.
- **`quant_4lvl` is non-monotone**: confidence rises 0.083 → 0.157 as the attack
  gets *harsher*, and the unwatermarked arm rises with it (0.041 → 0.139). Both
  stay far below 0.5, so no false positive — but the direction matches the 440 Hz
  tonal false positive in the null test, and 4-level quantisation does produce a
  near-square waveform. Same defect, not a new one.
- **`src_pesq` is non-monotone at the bottom** (`hp_2000hz` 1.53 vs `hp_2500hz`
  1.64). PESQ saturating, per Limit #5. Nothing below ~1.7 should be read as an
  ordering.

## Open question this run raises

At both failing cells the recovered payload sits above chance while confidence is
dead: median bit accuracy **0.65** at `hp_3200hz` (conf 0.135) and **0.70** at
`quant_8lvl` (conf 0.083), against a 0.50 floor. If that holds per clip, AWARE's
confidence head is giving up before the payload does. **Not claimed here** —
`summary.md` prints medians only, and confirming it needs the per-clip bit
accuracies from `data/raw.csv`.

## Limits (known before running)

1. **n=5.** No p-values will be reported: a paired sign test on 5 clips bottoms
   out at p=0.0625 and cannot reach 0.05 even if every clip agrees. The effect is
   large and mechanical or it is not there; `summary.md` prints every clip.
2. **No false-positive rate**, by construction — see Design.
3. **AWARE solo at 16 kHz**, as in the A/B. Production embeds through the full
   cascade at 22050 Hz with two resampling round-trips.
4. **Speech only**, ~10 s clips. Music is not covered here either.
5. **PESQ saturates near 1.0.** Below ~1.3 it cannot rank two destroyed files, so
   the listening files in `damage_ab/work/listen/` are part of the evidence, not
   a bonus.
6. **This says nothing about whether the failures should be fixed.** A destructive
   attack is still a real chain in the world (broadcast high-pass, low-bit
   telephony); "the audio was ruined anyway" is a scoping fact, not a dismissal.

## Reproduce

```bash
conda activate wmcompare
export WM_COMPARE_BASE=$HOME/wm_compare    # a login shell does not have this set
cd $WM_COMPARE_BASE/damage_ab
python sweep.py                            # checks anchors + ffmpeg
python make_pairs.py && python embed_pairs.py   # login node
sbatch damage_ab.sbatch
python analyze.py                          # after it finishes
```
