# Detection threshold — the standing decision and its evidence

**Current operating threshold: `conf >= 0.5` for AudioSeal and AWARE.**
**Last reviewed: 2026-08-15. Evidence: two runs, 740 scored decisions.**

This file exists because the threshold has been set four times by three people and
only the last of them was measured. Anyone proposing to change it should add a
row to the history below, with the run that justifies it.

## History

| Value | Set by | Basis | Outcome |
|---|---|---|---|
| `BIT_ACC_PRESENT = 0.8` | deployment layer | "reasonable" | ~100% false positives — the metric's null mean was 0.81 |
| `0.0007–0.0021` | `config.py` defaults | unknown | orders of magnitude below any real score |
| `0.65 / 0.55` | `for-prod-watermark` | observation that clean speech "often scores 0.4–0.55" | not reproduced; see below |
| **`0.5 / 0.5`** | upstream defaults | AudioSeal's documented default; AWARE's sigmoid midpoint (`x0 = 0.041`) | **confirmed by measurement** |

## The evidence

Two runs, designed independently, that answer different halves.

| Run | n | Answers |
|---|---|---|
| [2026-08-14_detector-null-test](2026-08-14_detector-null-test/) | 300 clean Emilia + 12 curated | false positives |
| [2026-08-10_aware-detection-ab](2026-08-10_aware-detection-ab/) | 20 wm + 20 clean × 11 conditions | true positives |

Together:

| | Result |
|---|---|
| Clean audio (n=300) | 0.0237 – **0.2723** |
| Watermarked, undistorted (n=20) | **0.9989** – 1.0000 |
| Separation | a gap of ~0.73, with 0.5 near its centre |
| False positives @ 0.5 | 0/300 → **< 1% at 95% confidence** (rule of three) |
| True positives @ 0.5 | 178/220 = **80.9%** across 11 conditions, AUC 0.9747 |

## The conflict between the two runs — and why 0.5 wins

The A/B derives a "calibrated" threshold of **0.2463**, being the lowest value
with zero false positives *on its 20 negatives*.

**The null test's 300 negatives reach 0.2723.** So 0.2463 is below the highest
clean score already observed, and adopting it would produce false positives on
audio we have already measured. This is precisely the A/B's own Limit #1 — 20
negatives cannot certify a low FPR — and neither run could have caught it alone.

The trade is poor regardless. Lowering 0.5 → 0.2463 gains 10 detections out of
220 (+4.6%), and **7 of those 10 fall in `highpass_0.2` and `quantize_8lvl`**,
which move from 0/20 and 1/20 to 4/20 each. Those conditions stay broken; the
cost is real false positives.

**Do not lower the threshold on the A/B's calibration.** If it is ever revisited,
it must be against negatives numbering in the hundreds, not twenty.

## What fails at 0.5, and what that means

| Condition | TPR@0.5 | PESQ | Reading |
|---|---|---|---|
| `highpass_0.2` | 0/20 | 1.31 | 1600 Hz cutoff removes part of AWARE's 1000–4000 Hz embedding band |
| `quantize_8lvl` | 1/20 | 1.04 | 3-bit requantisation; audio is destroyed |
| `time_stretch_1.1` | 17/20 | n/a | resampling desync, partial loss |
| pure 440 Hz tone | — | — | **false positive**, AWARE 0.9679 — tonal input, no threshold fixes it |

Both hard failures leave the audio at PESQ ≈ 1.0–1.3 against 4.29 clean, so the
file is barely listenable by the time detection fails. That is a mitigating fact,
not an excuse: broadcast and telephony chains do apply aggressive high-pass.

**Untested hypothesis:** these are AWARE-only failures. `bench_audioseal.csv` has
AudioSeal surviving high-pass and quantisation at comparable strengths, so the
cascade's OR rule may cover both gaps — which would be the first real evidence
that running two detectors buys robustness rather than only costing quality.
Different strengths and n=1 there, so this needs the cascade positive control
before it can be claimed.

## Scope — what these numbers do NOT cover

1. **The A/B measured AWARE solo at 16 kHz native.** Production embeds through
   the full cascade at 22050 Hz with two resampling round-trips. The null test
   used the cascade path; the A/B did not. Do not pool them without saying so.
2. **AudioSeal has no positive control at all.** Its clean distribution is
   measured (max 0.1497 over 300); its true-positive rate in the cascade is not.
3. **Speech only.** One 1.3 s music file across both runs. Customers upload music.
4. **One strength per attack.** Where a condition fails, neither run says how far
   from passing it was.

## Next review triggers

Change the threshold only if one of these happens:

- The cascade positive control shows TPR@0.5 materially below 80%
- Production logging shows real customer audio clustering above 0.27
- The tonal false positive turns out to reach realistic content (hold music,
  single instruments), which would call for an input guard rather than a
  threshold change
