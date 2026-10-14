# Dial sweep: AWARE tolerance vs CASP target (5 clips)

Clips: asv02, asv04, asv06, asv09, asv17. Same frozen detector. A = CASP audibility of the real watermark (1 = just noticeable). hard = mean over awgn_10, awgn_5, lp_2500, lp_2000, mp3_16, opus_12.

## Per dial point (mean over clips)

| family | dial | A_mean | A_max | pesq | stoi | snr_db | at_cap | clean | awgn_10 | awgn_5 | lp_2500 | lp_2000 | mp3_16 | opus_12 | hard_mean | hard_det |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AWARE | 6.000 | 7.642 | 17.018 | 4.086 | 0.987 | 17.105 | 0.000 | 1.000 | 0.970 | 0.800 | 0.880 | 0.800 | 0.980 | 0.860 | 0.882 | 0.600 |
| AWARE | 9.000 | 6.460 | 14.703 | 4.325 | 0.993 | 20.499 | 0.000 | 0.980 | 0.930 | 0.710 | 0.780 | 0.810 | 0.940 | 0.820 | 0.832 | 0.300 |
| AWARE | 12.000 | 5.390 | 12.947 | 4.459 | 0.997 | 22.606 | 0.000 | 0.950 | 0.840 | 0.680 | 0.730 | 0.760 | 0.820 | 0.800 | 0.772 | 0.100 |
| AWARE | 15.000 | 4.496 | 11.390 | 4.511 | 0.998 | 24.577 | 0.000 | 0.880 | 0.720 | 0.640 | 0.680 | 0.730 | 0.750 | 0.690 | 0.702 | 0.033 |
| AWARE | 18.000 | 3.667 | 9.620 | 4.562 | 0.999 | 26.271 | 0.000 | 0.840 | 0.690 | 0.610 | 0.660 | 0.700 | 0.720 | 0.680 | 0.677 | 0.033 |
| CASP | 1.000 | 1.519 | 2.317 | 4.643 | 1.000 | 32.981 | 0.017 | 0.640 | 0.550 | 0.560 | 0.570 | 0.610 | 0.590 | 0.490 | 0.562 | 0.000 |
| CASP | 2.000 | 2.648 | 3.790 | 4.635 | 0.999 | 28.910 | 0.156 | 0.740 | 0.600 | 0.590 | 0.620 | 0.650 | 0.660 | 0.590 | 0.618 | 0.000 |
| CASP | 4.000 | 3.825 | 5.937 | 4.605 | 0.998 | 25.693 | 0.656 | 0.920 | 0.720 | 0.610 | 0.650 | 0.720 | 0.760 | 0.690 | 0.692 | 0.000 |
| CASP | 6.000 | 4.220 | 7.368 | 4.587 | 0.997 | 24.569 | 0.782 | 0.910 | 0.770 | 0.630 | 0.670 | 0.730 | 0.810 | 0.700 | 0.718 | 0.000 |
| CASP | 8.000 | 4.564 | 8.220 | 4.559 | 0.997 | 24.280 | 0.822 | 0.910 | 0.770 | 0.650 | 0.700 | 0.720 | 0.830 | 0.700 | 0.728 | 0.000 |

## False positives on unwatermarked audio

| attack | fp |
|---|---|
| awgn_10 | 0/5 |
| awgn_5 | 0/5 |
| clean | 0/5 |
| lp_2000 | 0/5 |
| lp_2500 | 0/5 |
| mp3_16 | 0/5 |
| opus_12 | 0/5 |

## CASP family minus AWARE at the SAME audibility / SAME PESQ (hard-attack bit accuracy)

Only AWARE points that fall inside the CASP family's range are comparable (NaN otherwise).

| aware_tol | n_A | mean_delta_at_A | n_P | mean_delta_at_PESQ |
|---|---|---|---|---|
| 6.000 | 0.000 | nan | 0.000 | nan |
| 9.000 | 0.000 | nan | 0.000 | nan |
| 12.000 | 0.000 | nan | 0.000 | nan |
| 15.000 | 3.000 | -0.005 | 1.000 | -0.082 |
| 18.000 | 5.000 | -0.008 | 2.000 | -0.024 |

All comparable points: delta at equal A = -0.007 (n=8), at equal PESQ = -0.043 (n=3).

Positive = CASP-shaped budget keeps more bits. 20-bit payload: one bit = 0.05.
