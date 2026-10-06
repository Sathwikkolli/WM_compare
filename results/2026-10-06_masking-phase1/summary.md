# Phase 1 (E2–E6) summary — 2026-10-06_masking-phase1

> **Conclusion.** As a drop-in replacement, this masking model loses to AWARE's 6 dB rule on PESQ: at equal energy it is −0.25 PESQ worse; at equal PESQ it carries 5.6 dB less energy. It puts room in pauses (hearing floor + post-masking) and smears energy from harmonic peaks into valleys; tonality detection barely fires. At equal energy codecs keep it about as well as AWARE's shape. See README.

> **Gate 2: FAIL** — A (E3) ✗, B (E4) ✗, C (E6) ✗. Rule: (A or B) and C.

Clips: **30**. Shapes: `aware` = |X|·10^(−6/20); `mask` = Johnston-style masking threshold (`masking/psy.py`); `flat` = constant (control). Random ±1 signs on |X| in 1–4 kHz, phase kept (AWARE's watermark is ~signs × budget, E1a).

## E2 — where the two budgets differ

| quantity | median over clips | min | max |
|---|---|---|---|
| ratio_db_median | 4.591 | 1.636 | 9.365 |
| ratio_db_p10 | -4.000 | -5.227 | -1.608 |
| ratio_db_p90 | 16.319 | 11.376 | 22.638 |
| frac_mask_gt_aware | 0.750 | 0.613 | 0.856 |
| energy_ratio_db | -2.966 | -5.965 | -1.131 |
| ratio_db_quiet_frames | 8.607 | 0.859 | 16.642 |
| ratio_db_loud_frames | 3.120 | 2.144 | 4.697 |
| frac_bins_tonal | 0.001 | 0.000 | 0.017 |
| ratio_db_tonal | 9.935 | -14.568 | 15.468 |
| ratio_db_noisy | 4.564 | 1.636 | 9.378 |
| aware_above_mask | 0.250 | 0.144 | 0.387 |
| aware_above_mask_tonal | 0.292 | 0.000 | 0.972 |
| aware_above_mask_noisy | 0.250 | 0.144 | 0.387 |

ratio = mask budget / AWARE budget per bin (dB); `aware_above_mask` = share of bins where AWARE's change exceeds the modelled threshold.

## E3 — equal energy: which shape sounds better?

| shape | PESQ mean | PESQ median | STOI mean | Δ PESQ vs aware (median) | clips better | Wilcoxon p |
|---|---|---|---|---|---|---|
| aware | 4.129 | 4.123 | 0.995 | — | — | — |
| mask | 3.853 | 3.845 | 0.995 | -0.252 | 0/30 | 1.86e-09 |
| flat | 1.964 | 1.941 | 0.986 | -2.182 | 0/30 | 1.86e-09 |

**Gate 2A: fail** (median Δ -0.252, p = 1.86e-09).

## E4 — equal quality: extra energy the mask shape carries at AWARE's PESQ

| median | mean | IQR | min | max | clips > 0 dB |
|---|---|---|---|---|---|
| -5.64 dB | -5.59 dB | -7.11 … -4.04 | -11.09 | -1.99 | 0/30 |

Mask gain over its own threshold at that point: median -2.28 dB (0 dB = exactly at the modelled threshold).

**Gate 2B: fail** (median -5.64 dB; rule ≥ +1 dB).

## E5 — tonal vs noisy frames (equal energy within each frame type)

| frames | aware PESQ | mask PESQ | Δ median | clips better | p |
|---|---|---|---|---|---|
| noisy | 4.216 | 4.167 | -0.041 | 3/30 | 1.3e-07 |
| tonal | 4.507 | 4.348 | -0.157 | 0/30 | 1.86e-09 |

## E6 — codec survival of the change

| codec | shape | corr | kept (dB) | sign agree | sign agree (energy-wtd) |
|---|---|---|---|---|---|
| mp3_64 | aware | 0.983 | -0.31 | 0.825 | 0.968 |
| mp3_64 | mask_eq_energy | 0.978 | -0.30 | 0.897 | 0.978 |
| mp3_64 | mask_eq_quality | 0.939 | +0.09 | 0.839 | 0.947 |
| mp3_64 | flat | 0.962 | -0.31 | 0.969 | 0.993 |
| mp3_32 | aware | 0.759 | +1.67 | 0.662 | 0.824 |
| mp3_32 | mask_eq_energy | 0.697 | +2.16 | 0.748 | 0.836 |
| mp3_32 | mask_eq_quality | 0.537 | +4.46 | 0.667 | 0.762 |
| mp3_32 | flat | 0.612 | +1.51 | 0.895 | 0.945 |
| opus_32 | aware | 0.904 | +0.34 | 0.703 | 0.899 |
| opus_32 | mask_eq_energy | 0.871 | +0.24 | 0.790 | 0.916 |
| opus_32 | mask_eq_quality | 0.699 | +1.53 | 0.706 | 0.842 |
| opus_32 | flat | 0.783 | -0.38 | 0.929 | 0.972 |
| opus_16 | aware | 0.609 | +2.19 | 0.588 | 0.737 |
| opus_16 | mask_eq_energy | 0.512 | +2.67 | 0.660 | 0.745 |
| opus_16 | mask_eq_quality | 0.313 | +5.87 | 0.591 | 0.658 |
| opus_16 | flat | 0.376 | +1.93 | 0.875 | 0.925 |

corr = correlation between the change before and after the codec (1 = fully kept). kept = energy of the change after / before.

**Gate 2C: fail** (median corr ratio mask_eq_quality/aware: mp3_64 0.959, opus_32 0.772; rule ≥ 0.9 at both).

## Figures

- `figures/e2_budget_maps.png`
- `figures/e3_e4_e6.png`
