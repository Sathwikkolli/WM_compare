# VoxWatermark no-box benchmark — model comparison

Mean bit-accuracy across each attack's strength sweep. **Bold** = survives (mean bit-acc >= 0.8).

| Attack | AudioSeal | AWARE | Timbre |
|---|---|---|---|
| time_stretch | 0.72 | **0.97** | **1.00** |
| gaussian_noise | **0.93** | **0.99** | **0.96** |
| background_noise | **0.94** | **1.00** | **1.00** |
| opus | **1.00** | **1.00** | **1.00** |
| encodec | 0.79 | 0.75 | **0.84** |
| quantization | 0.76 | **0.89** | 0.78 |
| highpass | 0.53 | **0.83** | 0.72 |
| lowpass | **1.00** | **0.99** | **0.98** |
| smooth | **0.91** | **1.00** | **0.96** |
| echo | **1.00** | **1.00** | **1.00** |
| mp3 | **1.00** | **0.97** | **1.00** |
| aac | **1.00** | **1.00** | **1.00** |
| dynamic_compression | **1.00** | **1.00** | **1.00** |
| dynamic_expansion | **0.97** | **1.00** | **1.00** |
| inverse_polarity | 0.12 | **1.00** | **1.00** |
| time_jitter | **1.00** | **1.00** | **1.00** |
| phase_shift | **1.00** | **1.00** | **1.00** |
| **MEAN** | 0.86 | 0.96 | 0.96 |

## Overall robustness ranking

1. **AWARE** — mean bit-acc 0.964
2. **Timbre** — mean bit-acc 0.955
3. **AudioSeal** — mean bit-acc 0.863
