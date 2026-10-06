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

## Correction (found while reading the results)

The first `aggregate.py` printed **Gate 1a: SLACK** (9.3 % saturated). That was a
bug in `masking/common.py`, not a finding. On CPU, `aware.utils.to_tensor` returns
the same tensor and `.to("cpu")` is a no-op, so `AWAREEmbedder._optimize` updates
`initial_coeffs` **in place**. The capture read it *after* the call, so the
`initial` field in `data/e1a_<clip>.npz` and every number in `data/e1a_<clip>.json`
compare the best iterate with the *last* iterate, not with the original audio.

- Fixed: `OptimizeCapture` now copies before optimising.
- Recomputed without a rerun: `upper = M·(1 + 10^(−tol/20))` is unaffected, so
  `aggregate.py` rebuilds the true magnitudes from it. Corrected per-clip stats are
  in `e1a_stats.csv`. **The `data/e1a_<clip>.json` files are wrong** and are kept
  only because `data/` is never overwritten. Ignore `initial` in the npz files.
- The older single-clip probe (`probe_budget_utilization.py`) had the same bug, so
  its "~5 % hot tail" — the basis of prediction 3 — was also an artefact. Fixed too.
- E0 and E1b are unaffected (they never use the capture).

## Results (30 clips, 3.0–6.7 s, median 3.9 s)

**E0 — baseline to beat**

| | value |
|---|---|
| PESQ | 4.15 ± 0.20 (min 3.62) |
| STOI | 0.985 |
| SNR / SI-SNR | 17.1 ± 3.7 dB (min 9.9) / 18.5 dB |
| embed time (CPU, 4 cores) | median 12.2 s (one outlier 259 s) |
| clean bit accuracy | 0.997; 28/30 perfect, asv11 and asv18 have 1 wrong bit of 20 |
| 11 attacks | **identical to clean** for MP3 128/64, AWGN 25 dB, resample 8 k, lowpass 6 k, amp 0.6/1.4, requant 8 bit, crop 2048/6144 |
| median_3 | 0.983 (4 more clips lose 1–2 bits) |

**E1a — budget usage (corrected)**

| | value |
|---|---|
| coefficients at > 95 % of budget | **89.2 %** (82–94 % per clip) |
| median utilisation | **1.00** |
| quiet bins (bottom 25 %) at the limit | 99.1 % |
| loud bins (top 25 %) at the limit | 78.6 % |
| pushed up vs down at the limit | 48 % / 52 % |
| Spearman(frame energy, frame utilisation) | −0.76 |
| watermark energy relative to the band | −6.8 dB (the 6 dB tolerance itself) |

**E1b — window readability.** 1 s windows decode at chance (~0.5) below about
−30 dBFS and at 0.8–1.0 in speech (−25 to −12 dBFS). Q1→Q5 loudness: 0.50 → 0.90
on clean, and the **same curve** under mp3_64 and awgn_25.

## Gate 1

- **1a: BINDING** (89 % ≥ 25 %). Prediction 3 refuted (it rested on the buggy probe).
- **1b: errors CLUSTER** by loudness (Q5 − Q1 = +0.40 ≥ 0.10). Prediction 4 confirmed.

## What it means

1. **AWARE is a bang-bang embedder.** The optimiser pushes almost every 1–4 kHz
   coefficient to ±budget, about half up and half down. The watermark is therefore
   the budget times a sign pattern. **Whatever shape the budget has is the shape
   of the watermark**, so swapping the 6 dB rule for a masking threshold changes
   the watermark directly. This is the strongest possible case for E10.
2. **Quiet regions carry almost no watermark.** The budget is proportional to the
   magnitude, so pauses get a near-zero change and decode at chance even with no
   attack. Clean and attacked curves are the same, so this is an embedding
   limitation, not attack damage. Masking can add room here only through
   post-masking after loud frames and the hearing floor; frequency spreading helps
   inside speech frames, not in silence.
3. **The starter attacks are too weak to separate methods.** 11 of 12 give the
   clean BER. E10 (equal quality, compare BER) would show no difference with this
   set. **Before E10 we need a harder attack set** where baseline AWARE has
   BER around 5–20 %. Earlier runs here point to Encodec 6 kbps, MP3 8–16 kbps,
   resampling to 4 kHz, AWGN 5–10 dB, and highpass ~1–2 kHz.
4. Loud frames leave ~8 % of the budget unused — the optimiser stops needing it
   there. More room in loud frames is less valuable than more room elsewhere.

## Status

complete — E1a corrected in place of the first aggregate output (see Correction).
