# Summary — does the attack destroy the audio, or is AWARE fragile?

75 scored rows over 5 clips × 15 conditions.

Matched arms: every clip appears both unwatermarked (`src`) and AWARE-embedded (`wm`), so each row's arm comparison is within-clip.

## The pre-registered test

An attack is **destructive** when unwatermarked audio falls below PESQ 2.0 under it (`src_pesq`), and the watermarked arm is not worse by more than 0.3 PESQ. Both criteria and both constants were fixed in the run README before the run.

| condition | src PESQ (no wm) | wm PESQ | arm gap | wm conf | detected @0.5 | verdict |
|---|---|---|---|---|---|---|
| `clean` | 4.64 | 4.64 | 0.00 | 1.000 | 5/5 | survivable |
| `mp3_32k` | 3.76 | 3.78 | -0.06 | 0.995 | 5/5 | survivable |
| `hp_200hz` | 3.82 | 3.81 | 0.02 | 1.000 | 5/5 | survivable |
| `hp_500hz` | 2.45 | 2.44 | 0.01 | 1.000 | 5/5 | survivable |
| `hp_1000hz` | 1.81 | 1.85 | -0.03 | 1.000 | 5/5 | DESTRUCTIVE |
| `hp_1500hz` | 1.71 | 1.60 | -0.03 | 0.983 | 5/5 | DESTRUCTIVE |
| `hp_2000hz` | 1.53 | 1.63 | -0.02 | 0.800 | 5/5 | DESTRUCTIVE |
| `hp_2500hz` | 1.64 | 1.60 | 0.00 | 0.473 | 1/5 | DESTRUCTIVE |
| `hp_3200hz` | 1.32 | 1.37 | 0.01 | 0.135 | 0/5 | DESTRUCTIVE |
| `quant_256lvl` | 2.40 | 2.38 | -0.03 | 0.999 | 5/5 | survivable |
| `quant_64lvl` | 1.44 | 1.45 | -0.01 | 0.976 | 5/5 | DESTRUCTIVE |
| `quant_32lvl` | 1.17 | 1.18 | -0.00 | 0.866 | 5/5 | DESTRUCTIVE |
| `quant_16lvl` | 1.06 | 1.07 | -0.00 | 0.609 | 4/5 | DESTRUCTIVE |
| `quant_8lvl` | 1.04 | 1.04 | -0.00 | 0.083 | 0/5 | DESTRUCTIVE |
| `quant_4lvl` | 1.03 | 1.03 | -0.00 | 0.157 | 0/5 | DESTRUCTIVE |

`arm gap` = median per-clip (`src_pesq` − `wm_pesq`). Positive means the watermarked file was hurt more. Computed per clip and then aggregated, not as a difference of medians.

## The two cells the A/B reported as failures

**`hp_3200hz`** (highpass_0.2 — 0/20 in the A/B)

- unwatermarked audio under this attack: PESQ **1.32** (clean reference is the `clean` row's 4.19)
- watermarked audio under this attack: PESQ **1.37**, arm gap 0.01
- AWARE-band (1000–4000 Hz) energy surviving: **1.9%** of the watermarked signal's, 2.4% of the unwatermarked signal's
- detection: 0/5 at conf ≥ 0.5, median conf 0.135, median bit accuracy 0.650
- **verdict: DESTRUCTIVE**

**`quant_8lvl`** (quantize_8lvl — 1/20 in the A/B)

- unwatermarked audio under this attack: PESQ **1.04** (clean reference is the `clean` row's 4.19)
- watermarked audio under this attack: PESQ **1.04**, arm gap -0.00
- AWARE-band (1000–4000 Hz) energy surviving: **251.4%** of the watermarked signal's, 225.1% of the unwatermarked signal's
- detection: 0/5 at conf ≥ 0.5, median conf 0.083, median bit accuracy 0.700
- **verdict: DESTRUCTIVE**

## Where each attack breaks

- **highpass**: median confidence first drops below 0.5 at **2500 Hz** (conf 0.473, unwatermarked PESQ there 1.64, band kept 12.3%).
- **quantize**: median confidence first drops below 0.5 at **8 levels** (conf 0.083, unwatermarked PESQ there 1.04, band kept 251.4%).

Baseline for all of the above: the `clean` control detects 5/5 at conf ≥ 0.5 (median 1.000).

## Every condition

| condition | group | src PESQ | wm PESQ | comb PESQ | src SNR | wm SNR | src band | wm band | src conf | wm conf | bit acc |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `clean` | control | 4.64 | 4.64 | 4.19 | -- | -- | 100.0% | 100.0% | 0.097 | 1.000 | 1.000 |
| `mp3_32k` | anchor | 3.76 | 3.78 | 3.48 | 20.9 | 21.1 | 91.5% | 91.3% | 0.086 | 0.995 | 1.000 |
| `hp_200hz` | highpass | 3.82 | 3.81 | 3.71 | 8.9 | 8.8 | 100.0% | 100.0% | 0.096 | 1.000 | 1.000 |
| `hp_500hz` | highpass | 2.45 | 2.44 | 2.09 | 1.6 | 1.5 | 100.0% | 100.0% | 0.096 | 1.000 | 1.000 |
| `hp_1000hz` | highpass | 1.81 | 1.85 | 1.58 | 0.6 | 0.4 | 94.4% | 93.8% | 0.098 | 1.000 | 1.000 |
| `hp_1500hz` | highpass | 1.71 | 1.60 | 1.50 | 0.3 | 0.2 | 40.7% | 38.9% | 0.082 | 0.983 | 1.000 |
| `hp_2000hz` | highpass | 1.53 | 1.63 | 1.44 | 0.2 | 0.1 | 20.9% | 21.8% | 0.129 | 0.800 | 0.950 |
| `hp_2500hz` | highpass | 1.64 | 1.60 | 1.52 | 0.1 | 0.1 | 11.7% | 12.3% | 0.088 | 0.473 | 0.800 |
| `hp_3200hz` | highpass | 1.32 | 1.37 | 1.30 | 0.1 | 0.1 | 2.4% | 1.9% | 0.068 | 0.135 | 0.650 |
| `quant_256lvl` | quantize | 2.40 | 2.38 | 2.24 | 36.1 | 36.2 | 100.1% | 100.2% | 0.093 | 0.999 | 1.000 |
| `quant_64lvl` | quantize | 1.44 | 1.45 | 1.42 | 23.2 | 24.5 | 101.4% | 102.3% | 0.064 | 0.976 | 1.000 |
| `quant_32lvl` | quantize | 1.17 | 1.18 | 1.17 | 18.7 | 17.4 | 106.1% | 108.7% | 0.062 | 0.866 | 1.000 |
| `quant_16lvl` | quantize | 1.06 | 1.07 | 1.06 | 11.5 | 11.7 | 127.6% | 137.2% | 0.071 | 0.609 | 0.800 |
| `quant_8lvl` | quantize | 1.04 | 1.04 | 1.03 | 4.6 | 4.8 | 225.1% | 251.4% | 0.041 | 0.083 | 0.700 |
| `quant_4lvl` | quantize | 1.03 | 1.03 | 1.03 | -2.5 | -3.1 | 691.2% | 888.2% | 0.139 | 0.157 | 0.550 |

## Per-clip spread on the two failing cells

With n=5 the median alone can hide disagreement. These are the individual clips.

| condition | metric | cl00 | cl01 | cl02 | cl03 | cl04 |
|---|---|---|---|---|---|---|
| `hp_3200hz` | src_pesq | 1.75 | 1.37 | 1.32 | 1.29 | 1.12 |
| `hp_3200hz` | wm_pesq | 1.68 | 1.37 | 1.31 | 1.25 | 1.43 |
| `hp_3200hz` | wm_conf | 0.210 | 0.120 | 0.245 | 0.086 | 0.135 |
| `quant_8lvl` | src_pesq | 1.04 | 1.04 | 1.03 | 1.04 | 1.03 |
| `quant_8lvl` | wm_pesq | 1.05 | 1.04 | 1.03 | 1.04 | 1.03 |
| `quant_8lvl` | wm_conf | 0.048 | 0.067 | 0.083 | 0.215 | 0.453 |

## Limits

1. **n=5.** No p-values are reported: the smallest two-sided sign-test p reachable with 5 clips is 0.0625, which cannot reach 0.05 even if every clip agrees. Read the per-clip table above, not an interval.
2. **This run has no false-positive rate.** The `src` arm is the paired quality control, not an independent negative sample. FPR lives in `results/2026-08-14_detector-null-test/` (n=300).
3. **AWARE solo at 16 kHz**, as in the A/B. Production embeds through the full cascade at 22050 Hz with two resampling round-trips.
4. **Speech only**, ~10 s Emilia clips.
5. **PESQ is a speech-quality model.** At the bottom of its range it saturates near 1.0, so 'PESQ 1.04 vs 1.31' is not a meaningful ordering — both mean destroyed. Listen to `work/listen/` before quoting a difference down there.
