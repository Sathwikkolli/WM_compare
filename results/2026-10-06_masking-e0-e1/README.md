# 2026-10-06 — masking-e0-e1: AWARE baseline and budget usage

First step of the **masking-threshold-in-AWARE** project: replace AWARE's fixed
per-bin budget `|Δ| ≤ M·10^(−tol/20)` with a psychoacoustic masking threshold,
and test whether that lowers BER at equal quality. Before touching AWARE we need
(E0) the numbers to beat on our data, and (E1) whether the budget is what limits
AWARE at all.

Code: [`masking/`](../../masking/). Run order is in `masking/e0_e1.sbatch`.

## What

| ID | Question | Measured |
|---|---|---|
| **E0** | What does unmodified AWARE score on our clips? | PESQ, STOI, SNR, SI-SNR, ODG (if PEAQ installed), embed runtime; bit accuracy + confidence under 12 conditions |
| **E1a** | Does the optimiser use its budget? | utilisation = \|final − initial\| / budget per STFT coefficient; overall, quiet vs loud bins, per frequency, per frame |
| **E1b** | Where in the clip is the watermark readable? | 1 s windows (250 ms hop) decoded on their own, on clean / mp3_64 / awgn_25; window bit accuracy vs loudness and speech fraction |

E1b replaces "which frames fail" from Wang et al. 2025 (DSP 160:105025). AWARE
`full_length` carries one 20-bit message for the whole clip, so it has no
per-frame bit; decoding short windows is the closest equivalent.

## Setup

- **Audio:** 30 bona fide ASVspoof2019 LA eval utterances, one per speaker,
  ≥ 3 s, seed 0 (`masking/make_clips.py`). LibriSpeech is not on Great Lakes;
  LA eval bona fide is 16 kHz read speech (VCTK speakers), the closest match.
- **AWARE:** pip package at `fea9c49`, model `AWARE` (full_length):
  20 bits, 400 iterations, tolerance **6.0 dB**, band 1–4 kHz, NAdam lr 0.1.
  (The paper reports 6.2 dB / 500 iterations / 16 bps; we measure the code.)
- **Payload:** random per clip, `RandomState(clip_index)`.
- **Attacks:** clean, mp3_128, mp3_64, awgn_25, resample_8k, lowpass_6k,
  amp_0.6, amp_1.4, requant_8bit, median_3, crop_2048, crop_6144. Chosen to
  overlap Wang et al.'s common attacks (MP3 64/128, AWGN 25 dB, resampling,
  low-pass, amplitude scaling, requantisation, median filter, cropping) at 16 kHz.
- **No attacks inside embedding** (AWARE is attack-blind; we keep it that way).

## Gate 1 rules (fixed before the run)

- **1a — budget binding?** ≥ 25 % of coefficients at > 95 % utilisation →
  *binding*: a bigger masking budget can lower BER. < 10 % → *slack*: masking
  should mainly improve quality, not BER. In between → partly binding.
- **1b — do errors cluster?** If mean window bit accuracy differs by ≥ 0.10
  between the quietest and loudest loudness quintile, errors cluster, and
  frame-wise masking-capped boosting (E12b) is worth testing.

## Predictions (written before the run)

1. E0 clean: bit accuracy 1.0 on all clips; PESQ ≈ 4.2–4.4 (the cascade study
   measured 4.30 on Emilia).
2. E0 attacks: MP3, AWGN 25 dB, resampling, amplitude, requantisation ≈ 1.0;
   lowpass_6k ≈ 1.0 (band is 1–4 kHz); crops this small (128 / 384 ms) ≈ 1.0.
3. **E1a: slack.** An earlier single-clip probe (`probe_budget_utilization.py`,
   `analyze_utilization_2d.py`) saw a sparse hot tail of ~5 % high-utilisation
   coefficients. Expect < 10 % saturated, so masking's first benefit would be
   quality (E11) rather than BER (E10).
4. E1b: quiet / low-speech windows decode worse, especially under awgn_25.

**Caveat on 3:** low utilisation can also mean the loss is satisfied early,
not that room is useless. If 1a is slack, E10 still needs testing at a higher
operating point (lower tolerance), where the budget will bind.

## Outputs

- `data/e0_quality_<clip>.csv`, `data/e0_attacks_<clip>.csv`
- `data/e1a_<clip>.json` (stats) + `data/e1a_<clip>.npz` (initial / final / upper
  coefficients — reused later by E2 and E13)
- `data/e1b_<clip>.csv`
- `summary.md`, `e0_robustness_table.csv`, `params.json`, `figures/` — from `masking/aggregate.py`
- Audio (original + watermarked) stays on the cluster in `masking/work/` (gitignored).

## Status

planned — code committed, not yet run.
