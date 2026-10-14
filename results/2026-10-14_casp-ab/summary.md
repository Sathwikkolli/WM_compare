# Stock AWARE vs AWARE + CASP budget: 5 clips

Clips: asv02, asv04, asv06, asv09, asv17. Same frozen AWARE detector in both arms; detected = conf >= 0.5.

## Quality (watermarked vs original)

| arm | pesq_mean | pesq_std | stoi_mean | stoi_std | snr_db_mean | snr_db_std |
|---|---|---|---|---|---|---|
| AWARE | 4.068 | 0.164 | 0.987 | 0.002 | 16.958 | 1.928 |
| AWARE+CASP | 4.013 | 0.096 | 0.987 | 0.001 | 17.805 | 2.111 |

Per clip PESQ:

| clip | AWARE | AWARE+CASP | delta |
|---|---|---|---|
| asv02 | 4.334 | 4.144 | -0.190 |
| asv04 | 4.075 | 3.982 | -0.094 |
| asv06 | 4.050 | 4.014 | -0.036 |
| asv09 | 3.894 | 4.046 | 0.152 |
| asv17 | 3.987 | 3.880 | -0.107 |

## Budget actually used by arm B

| clip | kappa | eta_stock_median | eta_casp_median | frac_capped | E_ratio | align_corr |
|---|---|---|---|---|---|---|
| asv02 | 11.851 | 0.501 | 1.000 | 0.514 | 1.000 | 1.000 |
| asv04 | 13.973 | 0.501 | 0.416 | 0.142 | 1.000 | 1.000 |
| asv06 | 13.885 | 0.501 | 0.580 | 0.292 | 1.000 | 1.000 |
| asv09 | 18.515 | 0.501 | 1.000 | 0.501 | 1.000 | 1.000 |
| asv17 | 30.741 | 0.501 | 1.000 | 0.512 | 1.000 | 1.000 |

## Robustness (mean over clips)

| attack | AWARE bit_acc | AWARE det | AWARE+CASP bit_acc | AWARE+CASP det | CASP vs AWARE (W/T/L) | false pos (clean audio) |
|---|---|---|---|---|---|---|
| clean | 1.000 | 5/5 | 1.000 | 5/5 | 0/5/0 | 0/5 |
| mp3_64 | 1.000 | 5/5 | 1.000 | 5/5 | 0/5/0 | 0/5 |
| mp3_32 | 1.000 | 5/5 | 1.000 | 5/5 | 0/5/0 | 0/5 |
| awgn_20 | 1.000 | 5/5 | 0.990 | 5/5 | 0/4/1 | 0/5 |
| awgn_10 | 0.970 | 2/5 | 0.940 | 1/5 | 0/3/2 | 0/5 |
| lp_3500 | 1.000 | 5/5 | 1.000 | 5/5 | 0/5/0 | 0/5 |
| lp_2500 | 0.870 | 4/5 | 0.880 | 4/5 | 2/1/2 | 0/5 |

W/T/L = clips where AWARE+CASP bit accuracy is higher / equal / lower than stock AWARE.

## Per-clip bit accuracy

| attack | clip_id | AWARE | AWARE+CASP |
|---|---|---|---|
| clean | asv02 | 1.000 | 1.000 |
| clean | asv04 | 1.000 | 1.000 |
| clean | asv06 | 1.000 | 1.000 |
| clean | asv09 | 1.000 | 1.000 |
| clean | asv17 | 1.000 | 1.000 |
| mp3_64 | asv02 | 1.000 | 1.000 |
| mp3_64 | asv04 | 1.000 | 1.000 |
| mp3_64 | asv06 | 1.000 | 1.000 |
| mp3_64 | asv09 | 1.000 | 1.000 |
| mp3_64 | asv17 | 1.000 | 1.000 |
| mp3_32 | asv02 | 1.000 | 1.000 |
| mp3_32 | asv04 | 1.000 | 1.000 |
| mp3_32 | asv06 | 1.000 | 1.000 |
| mp3_32 | asv09 | 1.000 | 1.000 |
| mp3_32 | asv17 | 1.000 | 1.000 |
| awgn_20 | asv02 | 1.000 | 1.000 |
| awgn_20 | asv04 | 1.000 | 0.950 |
| awgn_20 | asv06 | 1.000 | 1.000 |
| awgn_20 | asv09 | 1.000 | 1.000 |
| awgn_20 | asv17 | 1.000 | 1.000 |
| awgn_10 | asv02 | 0.900 | 0.850 |
| awgn_10 | asv04 | 0.950 | 0.850 |
| awgn_10 | asv06 | 1.000 | 1.000 |
| awgn_10 | asv09 | 1.000 | 1.000 |
| awgn_10 | asv17 | 1.000 | 1.000 |
| lp_3500 | asv02 | 1.000 | 1.000 |
| lp_3500 | asv04 | 1.000 | 1.000 |
| lp_3500 | asv06 | 1.000 | 1.000 |
| lp_3500 | asv09 | 1.000 | 1.000 |
| lp_3500 | asv17 | 1.000 | 1.000 |
| lp_2500 | asv02 | 0.850 | 0.950 |
| lp_2500 | asv04 | 0.800 | 0.850 |
| lp_2500 | asv06 | 0.900 | 0.900 |
| lp_2500 | asv09 | 0.950 | 0.900 |
| lp_2500 | asv17 | 0.850 | 0.800 |
