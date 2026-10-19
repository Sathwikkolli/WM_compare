# MPEG-masking watermark vs AudioSeal / AWARE / Timbre on VoxWatermark attacks

Run `2026-10-19_mpeg-vox-bench`: **120** bona fide ASVspoof2019 LA eval clips (5 s crops), 66 attack configs from 17 VoxWatermark families.

Detection = payload test: at least k of n bits correct, k chosen so a random payload passes with ~1% probability.

| system | rule |
|---|---|
| MPEG | 13/16 |
| AudioSeal | 13/16 |
| AWARE | 16/20 |
| Timbre | 9/10 |

## Quality and clean decoding

| system | op_t | clips | pesq | pesq_sd | stoi | snr_db | scale | clamped | clean_bit_acc | clean_detect |
|---|---|---|---|---|---|---|---|---|---|---|
| MPEG | ship | 120 | 3.328 | 0.320 | 0.967 | 10.569 | 1.000 | 0 | 1.000 | 1.000 |
| MPEG | match@4.2 | 120 | 4.201 | 0.001 | 0.992 | 17.948 | 0.422 | 0 | 0.993 | 1.000 |
| AudioSeal | ship | 120 | 4.445 | 0.065 | 0.999 | 29.864 | 1.000 | 0 | 1.000 | 1.000 |
| AudioSeal | match@4.2 | 120 | 4.202 | 0.007 | 0.998 | 24.390 | 1.883 | 2 | 1.000 | 1.000 |
| AWARE | ship | 120 | 4.120 | 0.212 | 0.985 | 17.068 | 1.000 | 0 | 0.998 | 1.000 |
| AWARE | match@4.2 | 120 | 4.201 | 0.001 | 0.987 | 18.421 | 0.869 | 0 | 0.979 | 0.975 |
| Timbre | ship | 120 | 3.659 | 0.166 | 0.998 | 31.349 | 1.000 | 0 | 1.000 | 1.000 |
| Timbre | match@4.2 | 120 | 4.201 | 0.001 | 1.000 | 38.008 | 0.468 | 0 | 1.000 | 1.000 |

## False alarms (unwatermarked clips, decoded against the clip's payload)

| system | attacked | n | false_alarm | bit_acc |
|---|---|---|---|---|
| AWARE | False | 120 | 0.008 | 0.512 |
| AWARE | True | 7920 | 0.006 | 0.509 |
| AudioSeal | False | 120 | 0.000 | 0.471 |
| AudioSeal | True | 7920 | 0.005 | 0.479 |
| MPEG | False | 120 | 0.008 | 0.513 |
| MPEG | True | 7920 | 0.013 | 0.509 |
| Timbre | False | 120 | 0.000 | 0.484 |
| Timbre | True | 7920 | 0.007 | 0.495 |

## Per attack family, op = ship (mean bit acc / detection rate, all strengths pooled)

| family | MPEG | AudioSeal | AWARE | Timbre |
|---|---|---|---|---|
| time_stretch | 0.689 / 0.27 | 0.984 / 0.99 | 0.886 / 0.90 | 1.000 / 1.00 |
| gaussian_noise | 0.998 / 1.00 | 0.832 / 0.65 | 0.932 / 0.89 | 0.858 / 0.65 |
| background_noise | 1.000 / 1.00 | 0.853 / 0.69 | 0.962 / 0.96 | 0.983 / 0.96 |
| opus | 1.000 / 1.00 | 1.000 / 1.00 | 0.986 / 1.00 | 1.000 / 1.00 |
| encodec | 0.693 / 0.30 | 0.658 / 0.33 | 0.639 / 0.15 | 0.574 / 0.04 |
| quantization | 0.944 / 0.95 | 0.681 / 0.35 | 0.804 / 0.59 | 0.767 / 0.45 |
| highpass | 0.896 / 0.81 | 0.476 / 0.01 | 0.654 / 0.24 | 0.779 / 0.46 |
| lowpass | 0.936 / 0.86 | 1.000 / 1.00 | 0.952 / 0.90 | 0.980 / 0.96 |
| smooth | 0.648 / 0.18 | 0.863 / 0.75 | 0.962 / 0.99 | 0.976 / 0.98 |
| echo | 1.000 / 1.00 | 1.000 / 1.00 | 0.994 / 1.00 | 1.000 / 1.00 |
| mp3 | 0.898 / 0.79 | 0.996 / 1.00 | 0.901 / 0.80 | 0.984 / 0.96 |
| aac | 0.892 / 0.79 | 0.999 / 1.00 | 0.874 / 0.70 | 0.970 / 0.94 |
| dynamic_compression | 1.000 / 1.00 | 1.000 / 1.00 | 0.979 / 0.99 | 1.000 / 1.00 |
| dynamic_expansion | 0.999 / 1.00 | 0.989 / 1.00 | 0.994 / 1.00 | 1.000 / 1.00 |
| inverse_polarity | 0.000 / 0.00 | 0.331 / 0.00 | 0.998 / 1.00 | 1.000 / 1.00 |
| time_jitter | 0.999 / 1.00 | 1.000 / 1.00 | 0.986 / 1.00 | 1.000 / 1.00 |
| phase_shift | 1.000 / 1.00 | 1.000 / 1.00 | 0.998 / 1.00 | 1.000 / 1.00 |

## Per attack family, op = match@4.2 (mean bit acc / detection rate, all strengths pooled)

| family | MPEG | AudioSeal | AWARE | Timbre |
|---|---|---|---|---|
| time_stretch | 0.598 / 0.09 | 0.967 / 0.98 | 0.842 / 0.77 | 0.992 / 0.99 |
| gaussian_noise | 0.955 / 0.95 | 0.906 / 0.81 | 0.904 / 0.82 | 0.788 / 0.49 |
| background_noise | 0.984 / 1.00 | 0.921 / 0.87 | 0.937 / 0.91 | 0.923 / 0.79 |
| opus | 0.979 / 0.99 | 1.000 / 1.00 | 0.958 / 0.95 | 0.998 / 1.00 |
| encodec | 0.601 / 0.09 | 0.726 / 0.43 | 0.619 / 0.10 | 0.535 / 0.01 |
| quantization | 0.874 / 0.80 | 0.755 / 0.51 | 0.779 / 0.53 | 0.675 / 0.26 |
| highpass | 0.877 / 0.80 | 0.478 / 0.01 | 0.632 / 0.17 | 0.720 / 0.34 |
| lowpass | 0.888 / 0.78 | 1.000 / 1.00 | 0.930 / 0.86 | 0.970 / 0.92 |
| smooth | 0.578 / 0.06 | 0.889 / 0.79 | 0.924 / 0.91 | 0.911 / 0.76 |
| echo | 0.992 / 1.00 | 1.000 / 1.00 | 0.966 / 0.97 | 0.999 / 1.00 |
| mp3 | 0.841 / 0.70 | 0.999 / 1.00 | 0.871 / 0.76 | 0.973 / 0.92 |
| aac | 0.821 / 0.60 | 0.996 / 1.00 | 0.840 / 0.65 | 0.943 / 0.84 |
| dynamic_compression | 0.996 / 1.00 | 1.000 / 1.00 | 0.951 / 0.95 | 1.000 / 1.00 |
| dynamic_expansion | 0.962 / 0.99 | 0.998 / 1.00 | 0.967 / 0.96 | 0.995 / 1.00 |
| inverse_polarity | 0.007 / 0.00 | 0.366 / 0.00 | 0.979 / 0.97 | 1.000 / 1.00 |
| time_jitter | 0.996 / 1.00 | 1.000 / 1.00 | 0.956 / 0.95 | 1.000 / 1.00 |
| phase_shift | 0.993 / 1.00 | 1.000 / 1.00 | 0.979 / 0.97 | 1.000 / 1.00 |

Full per-strength numbers: `by_config.csv`. Figures: `fig_heatmap_*.png`, `fig_family_bars_*.png`.
