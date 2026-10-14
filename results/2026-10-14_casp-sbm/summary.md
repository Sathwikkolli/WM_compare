# AWARE vs CASP multibit, equal quality: 20 clips, 11 attack conditions

Quality axis for matching: **pesq** (every system matched to AWARE 6 dB's value on each clip). Detected = AWARE confidence >= 0.5, CASP/control chi-square p <= 1e-3.

## 1. Detection rate (share of clips where the watermark is found)

| system | clean | awgn_20 | awgn_10 | awgn_5 | lp_3500 | lp_2500 | lp_2000 | mp3_32 | mp3_16 | opus_12 | requant_6 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AWARE 6 dB (20 bits) | 1.00 | 0.95 | 0.10 | 0.05 | 1.00 | 0.50 | 0.25 | 1.00 | 1.00 | 0.65 | 1.00 | 0.68 |
| CASP 1 bit | 1.00 | 1.00 | 0.80 | 0.50 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.55 | 0.90 |
| CASP 4 bits | 1.00 | 1.00 | 0.85 | 0.40 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.45 | 0.88 |
| CASP 8 bits | 1.00 | 0.95 | 0.80 | 0.20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.45 | 0.85 |
| CASP 20 bits | 1.00 | 1.00 | 0.35 | 0.15 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.80 |
| AWARE-shape 20 bits (control) | 1.00 | 1.00 | 0.45 | 0.20 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.25 | 0.81 |

## 2. Bit accuracy (mean over clips; chance = 0.50)

| system | clean | awgn_20 | awgn_10 | awgn_5 | lp_3500 | lp_2500 | lp_2000 | mp3_32 | mp3_16 | opus_12 | requant_6 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AWARE 6 dB (20 bits) | 1.00 | 0.99 | 0.86 | 0.71 | 1.00 | 0.85 | 0.77 | 1.00 | 0.95 | 0.85 | 0.98 | 0.91 |
| CASP 1 bit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| CASP 4 bits | 1.00 | 1.00 | 0.99 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.99 |
| CASP 8 bits | 1.00 | 0.99 | 0.95 | 0.87 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.93 | 0.98 |
| CASP 20 bits | 1.00 | 0.97 | 0.85 | 0.73 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.81 | 0.94 |
| AWARE-shape 20 bits (control) | 1.00 | 0.97 | 0.85 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.82 | 0.94 |

## 3. Detection score (mean over clips; AWARE = confidence 0-1, others = z-equivalent, capped at 40)

| system | clean | awgn_20 | awgn_10 | awgn_5 | lp_3500 | lp_2500 | lp_2000 | mp3_32 | mp3_16 | opus_12 | requant_6 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AWARE 6 dB (20 bits) | 1.0 | 0.9 | 0.2 | 0.1 | 1.0 | 0.5 | 0.4 | 1.0 | 0.8 | 0.5 | 1.0 | 0.7 |
| CASP 1 bit | 34.0 | 10.1 | 4.9 | 3.1 | 34.0 | 33.4 | 30.6 | 29.4 | 18.0 | 14.4 | 4.0 | 19.6 |
| CASP 4 bits | 32.6 | 9.5 | 4.6 | 2.8 | 32.6 | 32.1 | 29.0 | 28.0 | 17.0 | 13.2 | 3.4 | 18.6 |
| CASP 8 bits | 32.2 | 8.7 | 4.1 | 2.3 | 32.1 | 31.6 | 28.2 | 27.4 | 15.9 | 13.1 | 3.0 | 18.1 |
| CASP 20 bits | 31.7 | 7.9 | 3.1 | 1.9 | 31.6 | 31.0 | 27.7 | 26.5 | 15.1 | 11.9 | 2.4 | 17.3 |
| AWARE-shape 20 bits (control) | 26.3 | 7.9 | 3.3 | 2.0 | 26.3 | 25.6 | 22.2 | 20.6 | 10.9 | 8.7 | 2.4 | 14.2 |

## 4. 20-bit head-to-head, per clip (difference = system minus AWARE; W/T/L = clips where it is better/equal/worse)

| attack | versus | bit_acc_diff | bit_W_T_L | det_diff | det_W_T_L |
|---|---|---|---|---|---|
| clean | CASP 20 bits | +0.000 | 0/20/0 | +0.000 | 0/20/0 |
| clean | AWARE-shape 20 bits (control) | +0.000 | 0/20/0 | +0.000 | 0/20/0 |
| awgn_20 | CASP 20 bits | -0.025 | 1/14/5 | +0.050 | 1/19/0 |
| awgn_20 | AWARE-shape 20 bits (control) | -0.025 | 2/10/8 | +0.050 | 1/19/0 |
| awgn_10 | CASP 20 bits | -0.008 | 8/5/7 | +0.250 | 6/13/1 |
| awgn_10 | AWARE-shape 20 bits (control) | -0.010 | 7/6/7 | +0.350 | 8/11/1 |
| awgn_5 | CASP 20 bits | +0.023 | 9/5/6 | +0.100 | 3/16/1 |
| awgn_5 | AWARE-shape 20 bits (control) | +0.037 | 10/3/7 | +0.150 | 3/17/0 |
| lp_3500 | CASP 20 bits | +0.000 | 0/20/0 | +0.000 | 0/20/0 |
| lp_3500 | AWARE-shape 20 bits (control) | +0.000 | 0/20/0 | +0.000 | 0/20/0 |
| lp_2500 | CASP 20 bits | +0.147 | 18/2/0 | +0.500 | 10/10/0 |
| lp_2500 | AWARE-shape 20 bits (control) | +0.147 | 18/2/0 | +0.500 | 10/10/0 |
| lp_2000 | CASP 20 bits | +0.230 | 20/0/0 | +0.750 | 15/5/0 |
| lp_2000 | AWARE-shape 20 bits (control) | +0.230 | 20/0/0 | +0.750 | 15/5/0 |
| mp3_32 | CASP 20 bits | +0.003 | 1/19/0 | +0.000 | 0/20/0 |
| mp3_32 | AWARE-shape 20 bits (control) | +0.003 | 1/19/0 | +0.000 | 0/20/0 |
| mp3_16 | CASP 20 bits | +0.050 | 15/5/0 | +0.000 | 0/20/0 |
| mp3_16 | AWARE-shape 20 bits (control) | +0.048 | 15/5/0 | +0.000 | 0/20/0 |
| opus_12 | CASP 20 bits | +0.148 | 19/1/0 | +0.350 | 7/13/0 |
| opus_12 | AWARE-shape 20 bits (control) | +0.150 | 19/1/0 | +0.350 | 7/13/0 |
| requant_6 | CASP 20 bits | -0.168 | 0/2/18 | -0.750 | 0/5/15 |
| requant_6 | AWARE-shape 20 bits (control) | -0.163 | 0/0/20 | -0.750 | 0/5/15 |

## 5. False positives on unwatermarked audio (count out of 20)

| system | clean | awgn_20 | awgn_10 | awgn_5 | lp_3500 | lp_2500 | lp_2000 | mp3_32 | mp3_16 | opus_12 | requant_6 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| AWARE 6 dB (20 bits) | 0 | 0 | 0 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| CASP 1 bit | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CASP 4 bits | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CASP 8 bits | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| CASP 20 bits | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

## 6. Matching diagnostics (g = strength, 1 = AWARE-6 dB energy; clamped = target quality not reachable inside g in [0.01, 2])

| system | pesq | stoi | snr_db | g | clamped_high | clamped_low |
|---|---|---|---|---|---|---|
| AWARE 6 dB (20 bits) | 4.150 | 0.986 | 17.655 |  | 0 | 0 |
| CASP 1 bit | 4.166 | 0.993 | 20.543 | 1.304 | 2 | 0 |
| CASP 4 bits | 4.174 | 0.993 | 20.299 | 1.394 | 3 | 0 |
| CASP 8 bits | 4.174 | 0.993 | 20.769 | 1.379 | 3 | 0 |
| CASP 20 bits | 4.176 | 0.993 | 20.918 | 1.396 | 3 | 0 |
| AWARE-shape 20 bits (control) | 4.152 | 0.995 | 22.665 | 1.048 | 0 | 0 |
