# E0 + E1 summary — 2026-10-06_masking-e0-e1

> **Conclusion.** Gate 1a: AWARE's budget is **binding** — 89 % of 1–4 kHz
> coefficients sit at the ±6 dB limit, so the budget's shape *is* the watermark's
> shape and a masking-threshold budget would change the watermark directly.
> Gate 1b: readability **clusters by loudness** — 1 s windows below ~−30 dBFS decode
> at chance, identically with and without attacks. The 12 starter attacks barely
> move BER (11 equal clean), so E10 needs a harder attack set first. E1a below is
> recomputed from the bounds; see README "Correction".

Clips: **30** bona fide ASVspoof2019 LA eval utterances (one per speaker). AWARE `AWARE`, 400 iterations, tolerance 6.0 dB, 20-bit payload.

## E0 — quality of the watermarked audio

| metric | mean ± std | median | min |
|---|---|---|---|
| pesq | 4.154 ± 0.196 | 4.135 | 3.623 |
| stoi | 0.985 ± 0.003 | 0.986 | 0.980 |
| snr_db | 17.122 ± 3.732 | 16.703 | 9.868 |
| si_snr_db | 18.487 ± 2.964 | 18.623 | 14.209 |
| odg | not measured | | |
| embed_seconds | 20.490 ± 44.996 | 12.195 | 10.170 |

## E0 — robustness

| attack | bit acc (mean ± std) | BER | clips perfect | clips ≥ 0.8 | conf mean |
|---|---|---|---|---|---|
| clean | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.999 |
| mp3_128 | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.999 |
| mp3_64 | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.999 |
| awgn_25 | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.982 |
| resample_8k | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.998 |
| lowpass_6k | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.999 |
| amp_0.6 | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.999 |
| amp_1.4 | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.999 |
| requant_8bit | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.999 |
| median_3 | 0.983 ± 0.040 | 0.017 | 80% | 100% | 0.971 |
| crop_2048 | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.999 |
| crop_6144 | 0.997 ± 0.013 | 0.003 | 93% | 100% | 0.999 |

## E1a — how much of AWARE's budget is used

| quantity | mean over clips | min | max |
|---|---|---|---|
| util_mean | 0.952 | 0.924 | 0.971 |
| util_median | 1.000 | 1.000 | 1.000 |
| frac_util_gt_0p95 | 0.892 | 0.820 | 0.938 |
| frac_util_gt_0p5 | 0.956 | 0.933 | 0.972 |
| frac_util_lt_0p05 | 0.004 | 0.003 | 0.007 |
| quiet_util_mean | 0.995 | 0.988 | 0.999 |
| loud_util_mean | 0.916 | 0.892 | 0.939 |
| quiet_frac_sat | 0.991 | 0.977 | 0.998 |
| loud_frac_sat | 0.786 | 0.730 | 0.859 |
| sat_pushed_up | 0.476 | 0.459 | 0.509 |
| spearman_frame_energy_vs_util | -0.761 | -0.894 | -0.450 |
| rel_change_energy_db | -6.802 | -7.721 | -6.407 |

**Gate 1a verdict: budget is BINDING** (89.2% of coefficients use > 95% of their budget; rule: ≥ 25% binding, < 10% slack).

## E1b — which windows carry the watermark

Window bit accuracy (1 s windows, 250 ms hop) by loudness quintile of the original audio (Q1 = quietest).

| condition | Q1 | Q2 | Q3 | Q4 | Q5 | Q5 − Q1 | Spearman(dBFS, acc) | speech<0.5 acc | speech≥0.5 acc |
|---|---|---|---|---|---|---|---|---|---|
| clean | 0.496 | 0.711 | 0.825 | 0.868 | 0.898 | +0.402 | 0.735 | 0.578 | 0.844 |
| mp3_64 | 0.501 | 0.706 | 0.823 | 0.871 | 0.901 | +0.401 | 0.737 | 0.583 | 0.843 |
| awgn_25 | 0.509 | 0.673 | 0.814 | 0.838 | 0.894 | +0.385 | 0.721 | 0.578 | 0.824 |

**Gate 1b verdict: errors CLUSTER by loudness** (largest |Q5 − Q1| = 0.402; rule: ≥ 0.10).

## Figures

- `figures/e1a_utilisation.png`
- `figures/e1b_window_acc_vs_loudness.png`
