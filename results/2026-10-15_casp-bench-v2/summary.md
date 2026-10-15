# SS vs AudioSeal / AWARE / Timbre

Clips per dataset: {'asv': 20, 'emilia': 20}. Payload test thresholds (1% binomial): AudioSeal 13/16, AWARE 16/20, Timbre 9/10, SS (AWARE shape) 13/16, SS (CASP shape) 13/16, SS (CASP shuffled, all) 13/16, SS (CASP shuffled, time) 13/16.

## Quality at each operating point (mean over clips)

| op | target | system | pesq | stoi | snr_db | scale |
|---|---|---|---|---|---|---|
| match | 4.2 | AudioSeal | 4.202 | 0.997 | 23.936 | 1.912 |
| match | 4.2 | AWARE | 4.201 | 0.986 | 17.831 | 0.966 |
| match | 4.2 | SS (AWARE shape) | 4.201 | 0.994 | 21.969 | 1.160 |
| match | 4.2 | SS (CASP shape) | 4.201 | 0.993 | 20.729 | 1.259 |
| match | 4.2 | SS (CASP shuffled, all) | 4.201 | 0.995 | 23.293 | 1.072 |
| match | 4.2 | SS (CASP shuffled, time) | 4.201 | 0.994 | 22.529 | 1.131 |
| match | 4.2 | Timbre | 4.201 | 0.999 | 36.762 | 0.581 |
| ship |  | AudioSeal | 4.444 | 0.999 | 29.364 | 1.000 |
| ship |  | AWARE | 4.165 | 0.986 | 17.246 | 1.000 |
| ship |  | SS (AWARE shape) | 4.167 | 0.994 | 21.293 | 1.149 |
| ship |  | SS (CASP shape) | 4.152 | 0.993 | 20.084 | 1.338 |
| ship |  | SS (CASP shuffled, all) | 4.152 | 0.995 | 21.974 | 1.117 |
| ship |  | SS (CASP shuffled, time) | 4.152 | 0.995 | 21.457 | 1.171 |
| ship |  | Timbre | 3.765 | 0.998 | 31.636 | 1.000 |

## Payload-test TPR at about 1% FPR, operating point `ship` : mean over attack strengths and clips

| family | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| lp_hz | 0.98 | 0.69 | 0.87 | 0.99 | 1.00 | 0.97 | 0.98 |
| vox_encodec | 0.40 | 0.17 | 0.07 | 0.24 | 0.14 | 0.09 | 0.04 |
| vox_gaussian_noise | 0.66 | 0.91 | 0.62 | 0.93 | 0.88 | 0.85 | 0.89 |
| vox_mp3 | 0.97 | 0.81 | 0.95 | 0.94 | 0.96 | 0.88 | 0.91 |
| vox_opus | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| requant | 0.82 | 0.93 | 0.83 | 0.83 | 0.78 | 0.77 | 0.72 |
| notch_worst | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_time_stretch | 0.96 | 0.94 | 0.99 | 0.11 | 0.23 | 0.05 | 0.12 |
| resample | 1.00 | 0.82 | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 |
| pitch | 0.02 | 0.01 | 0.01 | 0.03 | 0.03 | 0.00 | 0.01 |
| gain | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_aac | 1.00 | 0.86 | 0.94 | 1.00 | 1.00 | 0.97 | 1.00 |
| vox_dynamic_compression | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| notch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hp_hz | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| crop_head | 1.00 | 0.93 | 1.00 | 1.00 | 1.00 | 0.97 | 0.98 |
| vox_dynamic_expansion | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_echo | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_background_noise | 0.79 | 0.96 | 0.97 | 1.00 | 1.00 | 0.98 | 0.98 |
| vox_inverse_polarity | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_lowpass | 1.00 | 0.93 | 0.98 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_phase_shift | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_quantization | 0.43 | 0.65 | 0.49 | 0.51 | 0.38 | 0.24 | 0.27 |
| vox_highpass | 0.01 | 0.32 | 0.49 | 0.69 | 0.59 | 0.60 | 0.60 |
| vox_smooth | 0.77 | 0.99 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_time_jitter | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### lp_hz, `ship` 

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 4000Hz | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 3500Hz | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 3000Hz | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2500Hz | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2000Hz | 1.00 | 0.62 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1500Hz | 1.00 | 0.23 | 0.75 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1000Hz | 0.85 | 0.00 | 0.40 | 0.90 | 1.00 | 0.80 | 0.85 |

### vox_encodec, `ship` 

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 24.0kbps | 1.00 | 0.47 | 0.25 | 0.47 | 0.40 | 0.15 | 0.20 |
| 12.0kbps | 0.80 | 0.28 | 0.05 | 0.45 | 0.25 | 0.25 | 0.00 |
| 6.0kbps | 0.07 | 0.10 | 0.03 | 0.17 | 0.05 | 0.05 | 0.00 |
| 3.0kbps | 0.05 | 0.00 | 0.03 | 0.07 | 0.00 | 0.00 | 0.00 |
| 1.5kbps | 0.05 | 0.00 | 0.00 | 0.05 | 0.00 | 0.00 | 0.00 |

### vox_gaussian_noise, `ship` 

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 40dB | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 30dB | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 20dB | 0.93 | 1.00 | 0.90 | 1.00 | 1.00 | 1.00 | 1.00 |
| 10dB | 0.25 | 0.90 | 0.17 | 0.90 | 0.95 | 0.80 | 0.95 |
| 5dB | 0.10 | 0.65 | 0.05 | 0.72 | 0.45 | 0.45 | 0.50 |

### vox_mp3, `ship` 

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 40k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 32k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 24k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 16k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8k | 0.85 | 0.05 | 0.75 | 0.68 | 0.80 | 0.40 | 0.55 |

### vox_opus, `ship` 

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 256k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 128k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 64k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 32k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 16k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### requant, `ship` 

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 16bit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 12bit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8bit | 1.00 | 1.00 | 1.00 | 0.95 | 0.95 | 0.90 | 0.85 |
| 6bit | 0.97 | 1.00 | 0.97 | 0.80 | 0.70 | 0.70 | 0.70 |
| 4bit | 0.15 | 0.65 | 0.17 | 0.40 | 0.25 | 0.25 | 0.05 |

### notch_worst, `ship` 

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| worst | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### vox_time_stretch, `ship` 

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 0.9 | 0.95 | 1.00 | 1.00 | 0.10 | 0.30 | 0.00 | 0.10 |
| 1.1 | 1.00 | 1.00 | 1.00 | 0.23 | 0.55 | 0.10 | 0.15 |
| 0.7 | 0.93 | 0.88 | 1.00 | 0.10 | 0.00 | 0.05 | 0.05 |
| 1.3 | 0.97 | 0.97 | 1.00 | 0.05 | 0.20 | 0.05 | 0.25 |
| 1.5 | 0.95 | 0.85 | 0.97 | 0.07 | 0.10 | 0.05 | 0.05 |

### resample, `ship` 

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 16000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 11025 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4000 | 1.00 | 0.28 | 0.40 | 1.00 | 1.00 | 1.00 | 1.00 |

## Payload-test TPR at about 1% FPR, operating point `match` 4.2: mean over attack strengths and clips

| family | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| lp_hz | 1.00 | 0.66 | 0.84 | 0.97 | 0.99 | 0.97 | 0.96 |
| vox_encodec | 0.45 | 0.21 | 0.03 | 0.22 | 0.13 | 0.11 | 0.05 |
| vox_gaussian_noise | 0.84 | 0.89 | 0.54 | 0.90 | 0.84 | 0.80 | 0.83 |
| vox_mp3 | 1.00 | 0.80 | 0.93 | 0.91 | 0.95 | 0.87 | 0.89 |
| vox_opus | 1.00 | 0.98 | 1.00 | 1.00 | 1.00 | 0.99 | 1.00 |
| requant | 0.92 | 0.93 | 0.75 | 0.80 | 0.78 | 0.75 | 0.68 |
| notch_worst | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_time_stretch | 0.94 | 0.84 | 0.97 | 0.11 | 0.25 | 0.11 | 0.09 |
| resample | 1.00 | 0.78 | 0.79 | 1.00 | 1.00 | 0.97 | 0.97 |
| pitch | 0.02 | 0.01 | 0.01 | 0.02 | 0.00 | 0.01 | 0.00 |
| gain | 1.00 | 0.94 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_aac | 1.00 | 0.80 | 0.89 | 0.97 | 1.00 | 0.88 | 1.00 |
| vox_dynamic_compression | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| notch | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| hp_hz | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| crop_head | 1.00 | 0.92 | 1.00 | 0.98 | 1.00 | 0.92 | 0.93 |
| vox_dynamic_expansion | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_echo | 1.00 | 1.00 | 0.99 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_background_noise | 0.94 | 0.95 | 0.82 | 0.98 | 0.99 | 0.94 | 0.93 |
| vox_inverse_polarity | 0.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_lowpass | 1.00 | 0.91 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_phase_shift | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_quantization | 0.61 | 0.64 | 0.31 | 0.49 | 0.30 | 0.25 | 0.22 |
| vox_highpass | 0.01 | 0.33 | 0.41 | 0.67 | 0.61 | 0.61 | 0.62 |
| vox_smooth | 0.80 | 0.95 | 0.79 | 1.00 | 1.00 | 1.00 | 1.00 |
| vox_time_jitter | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

### lp_hz, `match` 4.2

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 4000Hz | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 3500Hz | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 3000Hz | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2500Hz | 1.00 | 0.85 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 2000Hz | 1.00 | 0.57 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| 1500Hz | 1.00 | 0.23 | 0.57 | 0.95 | 1.00 | 1.00 | 1.00 |
| 1000Hz | 0.97 | 0.00 | 0.30 | 0.88 | 0.95 | 0.80 | 0.75 |

### vox_encodec, `match` 4.2

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 24.0kbps | 1.00 | 0.47 | 0.07 | 0.40 | 0.35 | 0.40 | 0.15 |
| 12.0kbps | 0.97 | 0.38 | 0.03 | 0.42 | 0.20 | 0.10 | 0.10 |
| 6.0kbps | 0.23 | 0.20 | 0.00 | 0.20 | 0.05 | 0.05 | 0.00 |
| 3.0kbps | 0.05 | 0.03 | 0.00 | 0.05 | 0.00 | 0.00 | 0.00 |
| 1.5kbps | 0.00 | 0.00 | 0.03 | 0.03 | 0.05 | 0.00 | 0.00 |

### vox_gaussian_noise, `match` 4.2

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 40dB | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 30dB | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |
| 20dB | 1.00 | 0.97 | 0.62 | 0.97 | 0.95 | 0.90 | 0.90 |
| 10dB | 0.75 | 0.85 | 0.05 | 0.88 | 0.80 | 0.70 | 0.75 |
| 5dB | 0.45 | 0.62 | 0.05 | 0.62 | 0.45 | 0.40 | 0.50 |

### vox_mp3, `match` 4.2

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 40k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 32k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 24k | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 |
| 16k | 1.00 | 0.97 | 1.00 | 0.93 | 1.00 | 0.95 | 0.95 |
| 8k | 1.00 | 0.05 | 0.65 | 0.62 | 0.75 | 0.40 | 0.55 |

### vox_opus, `match` 4.2

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 256k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 128k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 64k | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 32k | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 16k | 1.00 | 0.95 | 1.00 | 1.00 | 1.00 | 0.95 | 1.00 |

### requant, `match` 4.2

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 16bit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 12bit | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8bit | 1.00 | 1.00 | 1.00 | 0.88 | 0.95 | 0.85 | 0.75 |
| 6bit | 1.00 | 1.00 | 0.70 | 0.80 | 0.75 | 0.60 | 0.60 |
| 4bit | 0.57 | 0.65 | 0.05 | 0.33 | 0.20 | 0.30 | 0.05 |

### notch_worst, `match` 4.2

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| worst | 1.00 | 1.00 | 0.97 | 1.00 | 1.00 | 1.00 | 1.00 |

### vox_time_stretch, `match` 4.2

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 0.9 | 0.95 | 0.93 | 1.00 | 0.23 | 0.40 | 0.00 | 0.20 |
| 1.1 | 1.00 | 0.88 | 1.00 | 0.25 | 0.55 | 0.40 | 0.20 |
| 0.7 | 0.93 | 0.88 | 1.00 | 0.05 | 0.05 | 0.00 | 0.00 |
| 1.3 | 0.95 | 0.88 | 0.93 | 0.03 | 0.15 | 0.10 | 0.00 |
| 1.5 | 0.88 | 0.68 | 0.93 | 0.00 | 0.10 | 0.05 | 0.05 |

### resample, `match` 4.2

| strength (mild to strong) | AudioSeal | AWARE | Timbre | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|---|
| 16000 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 11025 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| 8000 | 1.00 | 0.97 | 0.95 | 1.00 | 1.00 | 1.00 | 1.00 |
| 4000 | 1.00 | 0.15 | 0.23 | 1.00 | 1.00 | 0.90 | 0.90 |

## Breaking point: first strength where TPR falls below 0.5

| op | family | AWARE | AudioSeal | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) | Timbre |
|---|---|---|---|---|---|---|---|---|
| match 4.2 | crop_head | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | gain | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | hp_hz | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | lp_hz | 1500Hz | survives all | survives all | survives all | survives all | survives all | 1000Hz |
| match 4.2 | notch | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | notch_worst | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | pitch | -1 | -1 | -1 | -1 | -1 | -1 | -1 |
| match 4.2 | requant | survives all | survives all | 4bit | 4bit | 4bit | 4bit | 4bit |
| match 4.2 | resample | 4000 | survives all | survives all | survives all | survives all | survives all | 4000 |
| match 4.2 | vox_aac | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | vox_background_noise | survives all | survives all | survives all | survives all | survives all | survives all | 5dB |
| match 4.2 | vox_dynamic_compression | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | vox_dynamic_expansion | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | vox_echo | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | vox_encodec | 24.0kbps | 6.0kbps | 24.0kbps | 24.0kbps | 24.0kbps | 24.0kbps | 24.0kbps |
| match 4.2 | vox_gaussian_noise | survives all | 5dB | survives all | 5dB | 5dB | survives all | 10dB |
| match 4.2 | vox_highpass | 0.2 | 0.1 | 0.5 | 0.4 | 0.4 | 0.4 | 0.3 |
| match 4.2 | vox_inverse_polarity | survives all | neg | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | vox_lowpass | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | vox_mp3 | 8k | survives all | survives all | survives all | 8k | survives all | survives all |
| match 4.2 | vox_opus | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | vox_phase_shift | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | vox_quantization | 8lvl | 8lvl | 16lvl | 32lvl | 32lvl | 32lvl | 16lvl |
| match 4.2 | vox_smooth | survives all | w22 | survives all | survives all | survives all | survives all | w22 |
| match 4.2 | vox_time_jitter | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| match 4.2 | vox_time_stretch | survives all | survives all | 0.9 | 0.9 | 0.9 | 0.9 | survives all |
| ship  | crop_head | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | gain | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | hp_hz | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | lp_hz | 1500Hz | survives all | survives all | survives all | survives all | survives all | 1000Hz |
| ship  | notch | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | notch_worst | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | pitch | -1 | -1 | -1 | -1 | -1 | -1 | -1 |
| ship  | requant | survives all | 4bit | 4bit | 4bit | 4bit | 4bit | 4bit |
| ship  | resample | 4000 | survives all | survives all | survives all | survives all | survives all | 4000 |
| ship  | vox_aac | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_background_noise | survives all | 5dB | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_dynamic_compression | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_dynamic_expansion | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_echo | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_encodec | 24.0kbps | 6.0kbps | 24.0kbps | 24.0kbps | 24.0kbps | 24.0kbps | 24.0kbps |
| ship  | vox_gaussian_noise | survives all | 10dB | survives all | 5dB | 5dB | survives all | 10dB |
| ship  | vox_highpass | 0.2 | 0.1 | 0.5 | 0.4 | 0.4 | 0.4 | 0.4 |
| ship  | vox_inverse_polarity | survives all | neg | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_lowpass | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_mp3 | 8k | survives all | survives all | survives all | 8k | survives all | survives all |
| ship  | vox_opus | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_phase_shift | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_quantization | 8lvl | 16lvl | 8lvl | 8lvl | 32lvl | 32lvl | 16lvl |
| ship  | vox_smooth | survives all | w22 | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_time_jitter | survives all | survives all | survives all | survives all | survives all | survives all | survives all |
| ship  | vox_time_stretch | survives all | survives all | 0.9 | 0.9 | 0.9 | 0.9 | survives all |

## SS minus each neural system, mean TPR over all attacks (bootstrap 95% over clips)

| op | versus | mean_diff | ci_lo | ci_hi | clips_ss_better | n |
|---|---|---|---|---|---|---|
| ship  | AudioSeal | +0.032 | +0.022 | +0.044 | 30 | 40 |
| ship  | AWARE | +0.016 | +0.003 | +0.026 | 29 | 40 |
| ship  | Timbre | +0.009 | -0.004 | +0.021 | 21 | 40 |
| match 4.2 | AudioSeal | -0.013 | -0.029 | +0.001 | 15 | 40 |
| match 4.2 | AWARE | +0.020 | +0.004 | +0.033 | 29 | 40 |
| match 4.2 | Timbre | +0.048 | +0.034 | +0.062 | 34 | 40 |

## Is it CASP's map? SS (CASP shape) minus each control, mean TPR over all attacks (paired, bootstrap 95%)

Positive = the real CASP map is better than the control. A shuffled control that matches CASP means only the unevenness of the shape matters, not where CASP puts it.

| op | control | diff | ci_lo | ci_hi | clips_casp_better | n |
|---|---|---|---|---|---|---|
| ship  | SS (AWARE shape) | +0.012 | +0.000 | +0.023 | 12 | 20 |
| ship  | SS (CASP shuffled, all) | +0.029 | +0.010 | +0.046 | 15 | 20 |
| ship  | SS (CASP shuffled, time) | +0.023 | +0.012 | +0.034 | 16 | 20 |
| match 4.2 | SS (AWARE shape) | +0.023 | +0.003 | +0.045 | 12 | 20 |
| match 4.2 | SS (CASP shuffled, all) | +0.027 | +0.010 | +0.044 | 13 | 20 |
| match 4.2 | SS (CASP shuffled, time) | +0.031 | +0.016 | +0.047 | 17 | 20 |

### Attack families where CASP differs from a control with a 95% interval excluding 0 (matched point)

| control | family | diff | ci_lo | ci_hi | n |
|---|---|---|---|---|---|
| SS (AWARE shape) | vox_mp3 | +0.070 | +0.020 | +0.130 | 20 |
| SS (AWARE shape) | vox_time_stretch | +0.140 | +0.040 | +0.230 | 20 |
| SS (CASP shuffled, all) | vox_aac | +0.125 | +0.050 | +0.225 | 20 |
| SS (CASP shuffled, all) | vox_background_noise | +0.050 | +0.010 | +0.100 | 20 |
| SS (CASP shuffled, all) | vox_mp3 | +0.080 | +0.040 | +0.120 | 20 |
| SS (CASP shuffled, all) | vox_time_stretch | +0.140 | +0.060 | +0.220 | 20 |
| SS (CASP shuffled, time) | requant | +0.100 | +0.030 | +0.170 | 20 |
| SS (CASP shuffled, time) | vox_background_noise | +0.060 | +0.000 | +0.130 | 20 |
| SS (CASP shuffled, time) | vox_mp3 | +0.060 | +0.010 | +0.110 | 20 |
| SS (CASP shuffled, time) | vox_quantization | +0.080 | +0.030 | +0.130 | 20 |
| SS (CASP shuffled, time) | vox_time_stretch | +0.160 | +0.070 | +0.240 | 20 |

## False alarms: payload test on unwatermarked, attacked clips (random payloads)

| system | null_fpr | n |
|---|---|---|
| AudioSeal | 0.0108 | 515000 |
| AWARE | 0.0061 | 515000 |
| SS (AWARE shape) | 0.0106 | 515000 |
| Timbre | 0.0108 | 515000 |

## Native presence score, TPR at 1% FPR from the null (matched point; Timbre has none)

| family | AudioSeal | AWARE | SS (AWARE shape) | SS (CASP shape) | SS (CASP shuffled, all) | SS (CASP shuffled, time) |
|---|---|---|---|---|---|---|
| crop_head | 1.00 | 0.91 | 0.94 |  |  |  |
| gain | 1.00 | 0.97 | 1.00 |  |  |  |
| hp_hz | 1.00 | 0.99 | 1.00 |  |  |  |
| lp_hz | 1.00 | 0.72 | 0.97 |  |  |  |
| notch | 1.00 | 0.99 | 1.00 |  |  |  |
| notch_worst |  |  |  |  |  |  |
| pitch | 0.05 | 0.17 | 0.03 |  |  |  |
| requant | 0.81 | 0.93 | 0.74 |  |  |  |
| resample | 1.00 | 0.77 | 0.97 |  |  |  |
| vox_aac | 1.00 | 0.76 | 0.97 |  |  |  |
| vox_background_noise | 1.00 | 0.93 | 0.96 |  |  |  |
| vox_dynamic_compression | 1.00 | 0.99 | 1.00 |  |  |  |
| vox_dynamic_expansion | 1.00 | 0.97 | 0.97 |  |  |  |
| vox_echo | 1.00 | 0.97 | 1.00 |  |  |  |
| vox_encodec | 0.86 | 0.23 | 0.08 |  |  |  |
| vox_gaussian_noise | 0.99 | 0.85 | 0.88 |  |  |  |
| vox_highpass | 0.06 | 0.29 | 0.65 |  |  |  |
| vox_inverse_polarity | 1.00 | 1.00 | 1.00 |  |  |  |
| vox_lowpass | 1.00 | 0.93 | 1.00 |  |  |  |
| vox_mp3 | 0.99 | 0.83 | 0.88 |  |  |  |
| vox_opus | 1.00 | 0.98 | 0.99 |  |  |  |
| vox_phase_shift | 1.00 | 1.00 | 1.00 |  |  |  |
| vox_quantization | 0.58 | 0.69 | 0.44 |  |  |  |
| vox_smooth | 0.93 | 0.97 | 1.00 |  |  |  |
| vox_time_jitter | 1.00 | 0.99 | 0.99 |  |  |  |
| vox_time_stretch | 0.99 | 0.91 | 0.02 |  |  |  |
