# E0 + E1 summary — 2026-10-06_masking-e0-e1

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
| util_mean | 0.405 | 0.000 | 1.263 |
| util_median | 0.000 | 0.000 | 0.000 |
| frac_util_gt_0p95 | 0.093 | 0.000 | 0.172 |
| frac_util_gt_0p5 | 0.112 | 0.000 | 0.194 |
| frac_util_lt_0p05 | 0.831 | 0.735 | 1.000 |
| quiet_util_mean | 0.177 | 0.000 | 0.389 |
| loud_util_mean | 0.641 | 0.000 | 3.841 |
| quiet_frac_sat | 0.160 | 0.000 | 0.382 |
| loud_frac_sat | 0.070 | 0.000 | 0.109 |
| sat_pushed_up | 0.838 | 0.707 | 0.921 |
| spearman_frame_energy_vs_util | 0.085 | -0.277 | 0.634 |
| rel_change_energy_db | -inf | -inf | -24.575 |

**Gate 1a verdict: budget is SLACK** (9.3% of coefficients use > 95% of their budget; rule: ≥ 25% binding, < 10% slack).

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
