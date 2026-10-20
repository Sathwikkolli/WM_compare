# MPEG-masking watermark vs AudioSeal / AWARE / Timbre on VoxWatermark attacks

Runs `2026-10-20_casp-vox-bench`, `2026-10-19_mpeg-vox-bench`: **120** bona fide ASVspoof2019 LA eval clips (5 s crops), 66 attack configs from 17 VoxWatermark families.

Detection = payload test: at least k of n bits correct, k chosen so a random payload passes with ~1% probability.

| system | rule |
|---|---|
| MPEG | 13/16 |
| AudioSeal | 13/16 |
| AWARE | 16/20 |
| Timbre | 9/10 |
| CASP-1 | 13/16 |
| CASP-1-shuf | 13/16 |
| CASP-2 | 13/16 |
| CASP-2-shuf | 13/16 |

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
| CASP-1 | ship | 119 | 3.542 | 0.370 | 0.977 | 10.572 | 1.000 | 0 | 0.994 | 1.000 |
| CASP-1 | match@4.2 | 119 | 4.201 | 0.001 | 0.993 | 17.026 | 0.469 | 0 | 0.924 | 0.908 |
| CASP-1-shuf | ship | 119 | 3.277 | 0.308 | 0.967 | 10.586 | 1.000 | 0 | 0.997 | 1.000 |
| CASP-1-shuf | match@4.2 | 119 | 4.201 | 0.001 | 0.993 | 18.992 | 0.383 | 0 | 0.967 | 0.958 |
| CASP-2 | ship | 119 | 1.802 | 0.253 | 0.944 | 10.838 | 1.000 | 0 | 1.000 | 1.000 |
| CASP-2 | match@4.2 | 119 | 4.181 | 0.057 | 0.999 | 32.897 | 0.071 | 23 | 0.965 | 1.000 |
| CASP-2-shuf | ship | 119 | 1.439 | 0.138 | 0.952 | 11.185 | 1.000 | 0 | 1.000 | 1.000 |
| CASP-2-shuf | match@4.2 | 119 | 3.418 | 0.238 | 1.000 | 37.205 | 0.050 | 119 | 0.995 | 1.000 |

## False alarms (unwatermarked clips, decoded against the clip's payload)

| system | attacked | n | false_alarm | bit_acc |
|---|---|---|---|---|
| AWARE | False | 120 | 0.008 | 0.512 |
| AWARE | True | 7920 | 0.006 | 0.509 |
| AudioSeal | False | 120 | 0.000 | 0.471 |
| AudioSeal | True | 7920 | 0.005 | 0.479 |
| CASP-1 | False | 119 | 0.008 | 0.514 |
| CASP-1 | True | 7854 | 0.012 | 0.509 |
| CASP-1-shuf | False | 119 | 0.025 | 0.503 |
| CASP-1-shuf | True | 7854 | 0.013 | 0.498 |
| CASP-2 | False | 119 | 0.008 | 0.527 |
| CASP-2 | True | 7854 | 0.013 | 0.516 |
| CASP-2-shuf | False | 119 | 0.017 | 0.501 |
| CASP-2-shuf | True | 7854 | 0.013 | 0.497 |
| MPEG | False | 120 | 0.008 | 0.513 |
| MPEG | True | 7920 | 0.013 | 0.509 |
| Timbre | False | 120 | 0.000 | 0.484 |
| Timbre | True | 7920 | 0.007 | 0.495 |

## Paired comparisons (by clip, 95% bootstrap interval)

`detect_diff`: detection rate over all attack configs, system minus `versus`. `pesq_diff`: PESQ of the clean watermarked clip. At `ship` the CASP systems have exactly MPEG's watermark energy, so any difference there is placement.

| op | system | versus | detect_diff | ci | clips_better | pesq_diff | pesq_ci |
|---|---|---|---|---|---|---|---|
| ship | CASP-1 | MPEG | -0.050 | [-0.058, -0.043] | 11/119 | 0.213 | [+0.176, +0.250] |
| ship | CASP-2 | MPEG | 0.061 | [+0.051, +0.071] | 95/119 | -1.526 | [-1.582, -1.470] |
| ship | CASP-1 | CASP-1-shuf | -0.027 | [-0.040, -0.014] | 32/119 | 0.265 | [+0.216, +0.316] |
| ship | CASP-2 | CASP-2-shuf | 0.012 | [+0.002, +0.021] | 65/119 | 0.363 | [+0.316, +0.411] |
| ship | CASP-1 | CASP-2 | -0.111 | [-0.122, -0.100] | 3/119 | 1.739 | [+1.679, +1.800] |
| match@4.2 | CASP-1 | MPEG | -0.115 | [-0.140, -0.092] | 11/119 | -0.000 | [-0.000, +0.000] |
| match@4.2 | CASP-2 | MPEG | -0.112 | [-0.129, -0.096] | 10/119 | -0.020 | [-0.031, -0.011] |
| match@4.2 | CASP-1 | CASP-1-shuf | -0.055 | [-0.088, -0.022] | 40/119 | 0.000 | [-0.000, +0.000] |
| match@4.2 | CASP-2 | CASP-2-shuf | -0.044 | [-0.059, -0.029] | 31/119 | 0.763 | [+0.720, +0.808] |
| match@4.2 | CASP-1 | CASP-2 | -0.002 | [-0.033, +0.026] | 66/119 | 0.020 | [+0.011, +0.031] |

### Attack families where the interval excludes 0

| op | system | versus | family | detect_diff | ci |
|---|---|---|---|---|---|
| ship | CASP-1 | MPEG | aac | -0.101 | [-0.143, -0.055] |
| ship | CASP-1 | MPEG | background_noise | -0.015 | [-0.034, -0.003] |
| ship | CASP-1 | MPEG | encodec | -0.148 | [-0.187, -0.109] |
| ship | CASP-1 | MPEG | gaussian_noise | -0.042 | [-0.064, -0.022] |
| ship | CASP-1 | MPEG | highpass | -0.015 | [-0.029, -0.005] |
| ship | CASP-1 | MPEG | lowpass | -0.040 | [-0.059, -0.024] |
| ship | CASP-1 | MPEG | mp3 | -0.059 | [-0.079, -0.039] |
| ship | CASP-1 | MPEG | quantization | -0.091 | [-0.128, -0.054] |
| ship | CASP-1 | MPEG | smooth | -0.059 | [-0.087, -0.030] |
| ship | CASP-1 | MPEG | time_stretch | -0.143 | [-0.180, -0.104] |
| ship | CASP-2 | MPEG | encodec | 0.198 | [+0.150, +0.247] |
| ship | CASP-2 | MPEG | lowpass | 0.037 | [+0.013, +0.059] |
| ship | CASP-2 | MPEG | mp3 | 0.027 | [+0.010, +0.044] |
| ship | CASP-2 | MPEG | smooth | 0.281 | [+0.225, +0.336] |
| ship | CASP-2 | MPEG | time_stretch | 0.222 | [+0.178, +0.266] |
| ship | CASP-1 | CASP-1-shuf | encodec | -0.089 | [-0.146, -0.030] |
| ship | CASP-1 | CASP-1-shuf | gaussian_noise | -0.029 | [-0.050, -0.008] |
| ship | CASP-1 | CASP-1-shuf | mp3 | -0.034 | [-0.061, -0.005] |
| ship | CASP-1 | CASP-1-shuf | quantization | -0.074 | [-0.118, -0.030] |
| ship | CASP-1 | CASP-1-shuf | time_stretch | -0.113 | [-0.151, -0.076] |
| ship | CASP-2 | CASP-2-shuf | highpass | 0.007 | [+0.002, +0.013] |
| ship | CASP-2 | CASP-2-shuf | quantization | 0.062 | [+0.035, +0.091] |
| ship | CASP-2 | CASP-2-shuf | smooth | 0.129 | [+0.074, +0.185] |
| ship | CASP-2 | CASP-2-shuf | time_stretch | -0.047 | [-0.089, -0.005] |
| ship | CASP-1 | CASP-2 | aac | -0.155 | [-0.210, -0.101] |
| ship | CASP-1 | CASP-2 | background_noise | -0.015 | [-0.034, -0.003] |
| ship | CASP-1 | CASP-2 | encodec | -0.346 | [-0.393, -0.299] |
| ship | CASP-1 | CASP-2 | gaussian_noise | -0.042 | [-0.064, -0.022] |
| ship | CASP-1 | CASP-2 | lowpass | -0.077 | [-0.099, -0.054] |
| ship | CASP-1 | CASP-2 | mp3 | -0.086 | [-0.113, -0.061] |
| ship | CASP-1 | CASP-2 | quantization | -0.111 | [-0.151, -0.072] |
| ship | CASP-1 | CASP-2 | smooth | -0.339 | [-0.393, -0.286] |
| ship | CASP-1 | CASP-2 | time_stretch | -0.365 | [-0.400, -0.329] |
| match@4.2 | CASP-1 | MPEG | aac | -0.105 | [-0.147, -0.067] |
| match@4.2 | CASP-1 | MPEG | background_noise | -0.166 | [-0.224, -0.116] |
| match@4.2 | CASP-1 | MPEG | dynamic_compression | -0.063 | [-0.105, -0.029] |
| match@4.2 | CASP-1 | MPEG | dynamic_expansion | -0.202 | [-0.273, -0.134] |
| match@4.2 | CASP-1 | MPEG | echo | -0.086 | [-0.126, -0.050] |
| match@4.2 | CASP-1 | MPEG | gaussian_noise | -0.202 | [-0.247, -0.160] |
| match@4.2 | CASP-1 | MPEG | highpass | -0.166 | [-0.205, -0.128] |
| match@4.2 | CASP-1 | MPEG | lowpass | -0.113 | [-0.150, -0.077] |
| match@4.2 | CASP-1 | MPEG | mp3 | -0.119 | [-0.150, -0.091] |
| match@4.2 | CASP-1 | MPEG | opus | -0.128 | [-0.175, -0.086] |
| match@4.2 | CASP-1 | MPEG | phase_shift | -0.092 | [-0.143, -0.042] |
| match@4.2 | CASP-1 | MPEG | quantization | -0.282 | [-0.343, -0.227] |
| match@4.2 | CASP-1 | MPEG | time_jitter | -0.050 | [-0.088, -0.021] |
| match@4.2 | CASP-1 | MPEG | time_stretch | -0.027 | [-0.054, -0.000] |
| match@4.2 | CASP-2 | MPEG | aac | -0.143 | [-0.202, -0.088] |
| match@4.2 | CASP-2 | MPEG | background_noise | -0.087 | [-0.118, -0.059] |
| match@4.2 | CASP-2 | MPEG | dynamic_expansion | -0.092 | [-0.143, -0.046] |
| match@4.2 | CASP-2 | MPEG | echo | -0.027 | [-0.044, -0.013] |
| match@4.2 | CASP-2 | MPEG | encodec | -0.044 | [-0.077, -0.010] |
| match@4.2 | CASP-2 | MPEG | gaussian_noise | -0.341 | [-0.393, -0.289] |
| match@4.2 | CASP-2 | MPEG | highpass | -0.024 | [-0.040, -0.008] |
| match@4.2 | CASP-2 | MPEG | lowpass | -0.101 | [-0.129, -0.072] |
| match@4.2 | CASP-2 | MPEG | mp3 | -0.163 | [-0.200, -0.126] |
| match@4.2 | CASP-2 | MPEG | opus | -0.092 | [-0.124, -0.062] |
| match@4.2 | CASP-2 | MPEG | quantization | -0.455 | [-0.513, -0.397] |
| match@4.2 | CASP-2 | MPEG | time_jitter | -0.038 | [-0.063, -0.017] |
| match@4.2 | CASP-1 | CASP-1-shuf | background_noise | -0.101 | [-0.168, -0.034] |
| match@4.2 | CASP-1 | CASP-1-shuf | dynamic_expansion | -0.092 | [-0.172, -0.008] |
| match@4.2 | CASP-1 | CASP-1-shuf | gaussian_noise | -0.106 | [-0.163, -0.049] |
| match@4.2 | CASP-1 | CASP-1-shuf | highpass | -0.116 | [-0.158, -0.076] |
| match@4.2 | CASP-1 | CASP-1-shuf | opus | -0.081 | [-0.133, -0.030] |
| match@4.2 | CASP-1 | CASP-1-shuf | quantization | -0.143 | [-0.213, -0.074] |
| match@4.2 | CASP-2 | CASP-2-shuf | aac | -0.084 | [-0.134, -0.034] |
| match@4.2 | CASP-2 | CASP-2-shuf | background_noise | -0.066 | [-0.097, -0.037] |
| match@4.2 | CASP-2 | CASP-2-shuf | dynamic_expansion | -0.063 | [-0.109, -0.017] |
| match@4.2 | CASP-2 | CASP-2-shuf | echo | -0.027 | [-0.044, -0.013] |
| match@4.2 | CASP-2 | CASP-2-shuf | highpass | 0.024 | [+0.003, +0.044] |
| match@4.2 | CASP-2 | CASP-2-shuf | lowpass | -0.081 | [-0.109, -0.052] |
| match@4.2 | CASP-2 | CASP-2-shuf | mp3 | -0.146 | [-0.183, -0.109] |
| match@4.2 | CASP-2 | CASP-2-shuf | opus | -0.097 | [-0.128, -0.069] |
| match@4.2 | CASP-2 | CASP-2-shuf | quantization | 0.106 | [+0.044, +0.168] |
| match@4.2 | CASP-2 | CASP-2-shuf | time_jitter | -0.038 | [-0.063, -0.017] |
| match@4.2 | CASP-2 | CASP-2-shuf | time_stretch | -0.198 | [-0.235, -0.161] |
| match@4.2 | CASP-1 | CASP-2 | background_noise | -0.079 | [-0.140, -0.022] |
| match@4.2 | CASP-1 | CASP-2 | dynamic_compression | -0.063 | [-0.105, -0.029] |
| match@4.2 | CASP-1 | CASP-2 | dynamic_expansion | -0.109 | [-0.193, -0.034] |
| match@4.2 | CASP-1 | CASP-2 | echo | -0.059 | [-0.101, -0.020] |
| match@4.2 | CASP-1 | CASP-2 | gaussian_noise | 0.139 | [+0.076, +0.200] |
| match@4.2 | CASP-1 | CASP-2 | highpass | -0.143 | [-0.183, -0.103] |
| match@4.2 | CASP-1 | CASP-2 | mp3 | 0.044 | [+0.002, +0.086] |
| match@4.2 | CASP-1 | CASP-2 | phase_shift | -0.092 | [-0.143, -0.042] |
| match@4.2 | CASP-1 | CASP-2 | quantization | 0.173 | [+0.103, +0.240] |

## Per attack family, op = ship (mean bit acc / detection rate, all strengths pooled)

| family | MPEG | CASP-1 | CASP-1-shuf | CASP-2 | CASP-2-shuf | AudioSeal | AWARE | Timbre |
|---|---|---|---|---|---|---|---|---|
| time_stretch | 0.689 / 0.27 | 0.611 / 0.12 | 0.664 / 0.24 | 0.776 / 0.49 | 0.790 / 0.54 | 0.984 / 0.99 | 0.886 / 0.90 | 1.000 / 1.00 |
| gaussian_noise | 0.998 / 1.00 | 0.964 / 0.96 | 0.988 / 0.99 | 0.999 / 1.00 | 1.000 / 1.00 | 0.832 / 0.65 | 0.932 / 0.89 | 0.858 / 0.65 |
| background_noise | 1.000 / 1.00 | 0.979 / 0.98 | 0.992 / 0.99 | 1.000 / 1.00 | 1.000 / 1.00 | 0.853 / 0.69 | 0.962 / 0.96 | 0.983 / 0.96 |
| opus | 1.000 / 1.00 | 0.984 / 0.99 | 0.995 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 0.986 / 1.00 | 1.000 / 1.00 |
| encodec | 0.693 / 0.30 | 0.640 / 0.16 | 0.668 / 0.25 | 0.766 / 0.50 | 0.765 / 0.51 | 0.658 / 0.33 | 0.639 / 0.15 | 0.574 / 0.04 |
| quantization | 0.944 / 0.95 | 0.902 / 0.86 | 0.937 / 0.93 | 0.961 / 0.97 | 0.939 / 0.91 | 0.681 / 0.35 | 0.804 / 0.59 | 0.767 / 0.45 |
| highpass | 0.896 / 0.81 | 0.873 / 0.79 | 0.896 / 0.79 | 0.897 / 0.80 | 0.886 / 0.79 | 0.476 / 0.01 | 0.654 / 0.24 | 0.779 / 0.46 |
| lowpass | 0.936 / 0.86 | 0.904 / 0.82 | 0.916 / 0.81 | 0.949 / 0.89 | 0.947 / 0.88 | 1.000 / 1.00 | 0.952 / 0.90 | 0.980 / 0.96 |
| smooth | 0.648 / 0.18 | 0.615 / 0.13 | 0.623 / 0.13 | 0.748 / 0.47 | 0.686 / 0.34 | 0.863 / 0.75 | 0.962 / 0.99 | 0.976 / 0.98 |
| echo | 1.000 / 1.00 | 0.991 / 1.00 | 0.996 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 0.994 / 1.00 | 1.000 / 1.00 |
| mp3 | 0.898 / 0.79 | 0.860 / 0.73 | 0.877 / 0.76 | 0.913 / 0.82 | 0.916 / 0.82 | 0.996 / 1.00 | 0.901 / 0.80 | 0.984 / 0.96 |
| aac | 0.892 / 0.79 | 0.850 / 0.69 | 0.858 / 0.68 | 0.918 / 0.84 | 0.918 / 0.85 | 0.999 / 1.00 | 0.874 / 0.70 | 0.970 / 0.94 |
| dynamic_compression | 1.000 / 1.00 | 0.996 / 1.00 | 0.997 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 0.979 / 0.99 | 1.000 / 1.00 |
| dynamic_expansion | 0.999 / 1.00 | 0.964 / 0.99 | 0.991 / 0.99 | 0.999 / 1.00 | 1.000 / 1.00 | 0.989 / 1.00 | 0.994 / 1.00 | 1.000 / 1.00 |
| inverse_polarity | 0.000 / 0.00 | 0.006 / 0.00 | 0.003 / 0.00 | 0.000 / 0.00 | 0.000 / 0.00 | 0.331 / 0.00 | 0.998 / 1.00 | 1.000 / 1.00 |
| time_jitter | 0.999 / 1.00 | 0.995 / 1.00 | 0.997 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 0.986 / 1.00 | 1.000 / 1.00 |
| phase_shift | 1.000 / 1.00 | 0.994 / 1.00 | 0.997 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 1.000 / 1.00 | 0.998 / 1.00 | 1.000 / 1.00 |

## Per attack family, op = match@4.2 (mean bit acc / detection rate, all strengths pooled)

| family | MPEG | CASP-1 | CASP-1-shuf | CASP-2 | CASP-2-shuf | AudioSeal | AWARE | Timbre |
|---|---|---|---|---|---|---|---|---|
| time_stretch | 0.598 / 0.09 | 0.566 / 0.06 | 0.584 / 0.07 | 0.565 / 0.06 | 0.680 / 0.26 | 0.967 / 0.98 | 0.842 / 0.77 | 0.992 / 0.99 |
| gaussian_noise | 0.955 / 0.95 | 0.864 / 0.75 | 0.910 / 0.86 | 0.817 / 0.61 | 0.832 / 0.65 | 0.906 / 0.81 | 0.904 / 0.82 | 0.788 / 0.49 |
| background_noise | 0.984 / 1.00 | 0.887 / 0.83 | 0.949 / 0.93 | 0.916 / 0.91 | 0.957 / 0.98 | 0.921 / 0.87 | 0.937 / 0.91 | 0.923 / 0.79 |
| opus | 0.979 / 0.99 | 0.904 / 0.87 | 0.949 / 0.95 | 0.930 / 0.90 | 0.985 / 1.00 | 1.000 / 1.00 | 0.958 / 0.95 | 0.998 / 1.00 |
| encodec | 0.601 / 0.09 | 0.585 / 0.06 | 0.577 / 0.06 | 0.546 / 0.04 | 0.540 / 0.03 | 0.726 / 0.43 | 0.619 / 0.10 | 0.535 / 0.01 |
| quantization | 0.874 / 0.80 | 0.785 / 0.52 | 0.828 / 0.66 | 0.712 / 0.34 | 0.659 / 0.24 | 0.755 / 0.51 | 0.779 / 0.53 | 0.675 / 0.26 |
| highpass | 0.877 / 0.80 | 0.802 / 0.63 | 0.860 / 0.75 | 0.862 / 0.78 | 0.858 / 0.75 | 0.478 / 0.01 | 0.632 / 0.17 | 0.720 / 0.34 |
| lowpass | 0.888 / 0.78 | 0.824 / 0.66 | 0.850 / 0.70 | 0.831 / 0.67 | 0.878 / 0.75 | 1.000 / 1.00 | 0.930 / 0.86 | 0.970 / 0.92 |
| smooth | 0.578 / 0.06 | 0.573 / 0.06 | 0.562 / 0.05 | 0.553 / 0.05 | 0.541 / 0.04 | 0.889 / 0.79 | 0.924 / 0.91 | 0.911 / 0.76 |
| echo | 0.992 / 1.00 | 0.926 / 0.91 | 0.966 / 0.95 | 0.953 / 0.97 | 0.991 / 1.00 | 1.000 / 1.00 | 0.966 / 0.97 | 0.999 / 1.00 |
| mp3 | 0.841 / 0.70 | 0.784 / 0.58 | 0.803 / 0.60 | 0.770 / 0.54 | 0.837 / 0.69 | 0.999 / 1.00 | 0.871 / 0.76 | 0.973 / 0.92 |
| aac | 0.821 / 0.60 | 0.778 / 0.50 | 0.786 / 0.51 | 0.745 / 0.46 | 0.809 / 0.55 | 0.996 / 1.00 | 0.840 / 0.65 | 0.943 / 0.84 |
| dynamic_compression | 0.996 / 1.00 | 0.943 / 0.94 | 0.977 / 0.97 | 0.970 / 1.00 | 0.996 / 1.00 | 1.000 / 1.00 | 0.951 / 0.95 | 1.000 / 1.00 |
| dynamic_expansion | 0.962 / 0.99 | 0.874 / 0.79 | 0.915 / 0.88 | 0.903 / 0.90 | 0.943 / 0.96 | 0.998 / 1.00 | 0.967 / 0.96 | 0.995 / 1.00 |
| inverse_polarity | 0.007 / 0.00 | 0.076 / 0.00 | 0.033 / 0.00 | 0.035 / 0.00 | 0.005 / 0.00 | 0.366 / 0.00 | 0.979 / 0.97 | 1.000 / 1.00 |
| time_jitter | 0.996 / 1.00 | 0.943 / 0.95 | 0.975 / 0.97 | 0.944 / 0.96 | 0.996 / 1.00 | 1.000 / 1.00 | 0.956 / 0.95 | 1.000 / 1.00 |
| phase_shift | 0.993 / 1.00 | 0.923 / 0.91 | 0.967 / 0.96 | 0.965 / 1.00 | 0.995 / 1.00 | 1.000 / 1.00 | 0.979 / 0.97 | 1.000 / 1.00 |

Full per-strength numbers: `by_config.csv`. Figures: `fig_heatmap_*.png`, `fig_family_bars_*.png`.
