# exp_v16 — Critically-sampled subband watermarking (exp1 vs exp2)

*All results in one place. Run 2026-08-03 on Great Lakes, Tesla V100-PCIE-16GB.*

---

## 1. The question

AWARE embeds by adversarially optimising a 20-bit mark into the STFT of a
signal, against a frozen randomly-initialised detector. `fsss/band_steer.py`
already lets it write into an arbitrary frequency band by moving the *audio*
onto AWARE's window with an affine map (frequency shift, then rational
resample). That map's own docstring concedes a cost:

> `host` has more samples than degrees of freedom (74667 vs 64000 for a 4 s clip
> at r = 7/6), so most of its spectrum is interpolation, not signal, and
> anything written there dies on the way back.

**exp1 removes that waste.** Split 0–8000 Hz into M uniform strips; strip *k* is
exactly `sr/2M` Hz wide, so keeping every M-th sample is alias-free and lands
the strip **critically sampled** — as many samples as degrees of freedom, not
one more. Every optimiser variable then points somewhere that survives the trip
home.

**Does removing the waste help?** Two readings, and the experiment separates
them:

- *it helps* — the optimiser was wasting effort on directions that get deleted
- *it does not* — the perceptual clamp already pinned those directions, and all
  criticality buys is speed

**Answer, from this run: it does not help. The overcomplete map is better.**
See §6.

---

## 2. Configs

Strip 2 of 10 = **1600–2400 Hz** (even, so it lands upright rather than
frequency-reversed). Guard band 100 Hz inward, so the filter passes 1700–2300.

|  | all cells writable | salient-gated (`librosa_flux`) |
|---|---|---|
| **CRITICAL** — decimate ×10 | `exp1a` | `exp1b` |
| **OVERCOMPLETE** — affine map 4/15 | `exp2a` | `exp2b` |
| **reference** | `stock` = plain AWARE, no strip, no chain | |

| pairing | isolates |
|---|---|
| exp1a vs exp2a | the **sampling** question |
| exp1a vs exp1b | the **gating** question, critical sampling |
| exp2a vs exp2b | the same gating question, overcomplete sampling |

Both salient arms build masks with the **same code**
(`fsss/salient_mapped.py`), differing only in the time scale their resampling
implies — `1/10` for exp1, `4/15` for exp2. Anything else differing between
them would be a confound rather than a result.

---

## 3. Files

| file | role |
|---|---|
| `fsss/band_critical.py` | critical-decimation DSP + T1–T6 smoke test |
| `fsss/chain_embed_critical.py` | exp1a/exp1b embedder, chain in the optimiser loop |
| `fsss/chain_embed_salient.py` | exp2b — the salient arm `chain_embed.py` deferred to "step 3" |
| `fsss/salient_mapped.py` | shared mask builder, anchors on the original mapped through a rational time scale |
| `fsss/exp_v16_critical_decimation.py` | the experiment |
| `fsss/viz_critical.py` | figures F1–F6 |
| `fsss/exp_v16_full.sbatch` | the GPU job that produced this |

Unchanged: `band_steer.py`, `band_steer_torch.py`, `chain_embed.py`,
`staircase.py`, `salient_region.py`, and the stock `aware` install.

---

## 4. Validation

### 4.1 DSP (`python -m fsss.band_critical`, synthetic signal)

| test | result | target |
|---|---|---|
| T1 split exactness | **329.8 dB** | > 200 |
| T2 decimation round trip | **49.8 dB** | > 40 |
| T3 pipeline null | **52.8 dB** | > 40 |
| T4 placement 1800/2000/2200 Hz → | **2000 / 4000 / 6000 Hz** | exact |
| T5 wrong factor (÷8) | **0.0 dB** | ≪ T2 |
| T6 gradient through chain | finite, max 1.1e+01 | non-zero |

**T5 is the alignment argument, measured.** Decimating by 8 instead of 10 puts
a fold at 2000 Hz, straight through the strip, and the strip aliases onto
itself — error energy equals signal energy. The 50 dB gap between the right
factor and a wrong one is why the decimation factor must equal the band count.

**T4 confirms the fold arithmetic *and* the relabel together.** The host is
measured against `sr` even though it is really `sr/10`, because that is exactly
what AWARE does: `torch.stft` never receives a sample rate and the detector's
mel bank was built once at 16 kHz. The relabel costs zero lines of code.

### 4.2 Chain (`verify()`, on Emilia speech)

| | exp1a / exp1b | exp2a |
|---|---|---|
| STFT round trip | 133.6 dB | 133.9 dB |
| chain null | 31.4 dB | 28.7 dB |
| decimation round trip | 33.5 dB | n/a |
| **budget gap** | **18.0 dB** | **16.2 dB** |
| host samples | 15,594 = slot dof 15,594 → **critically sampled** | 41,584 for ~22,000 dof → overcomplete |
| passband dof | 11,695 (25% shortfall = guard band, not overcompleteness) | — |
| host frames | **61** (stock: 610) | **163** |

Two things worth carrying forward:

- **Frame count.** Decimating by 10 divides the frame count by 10. The BRH
  readout head averages over frames, and that averaging is where crop/stretch
  robustness comes from. exp1 has 61 frames to average where stock has 610.
- **Filter sharpness.** At `numtaps=1023` the decimation round trip read only
  27.0 dB on speech (vs 49.8 on the synthetic test — the test signal had strong
  tones at the passband centre and flattered it). Sharpening to 4095 taps bought
  **+6.5 dB**. The limit is `|H|²` (analysis) disagreeing with `|H|⁴` (analysis
  + synthesis) inside the filter transitions; ~6 dB per 4× taps. Clean bit
  accuracy was 1.000 either way, so ~30 dB is demonstrably sufficient.

---

## 5. Results — 3 Emilia clips × 10 s, tol 6, 400 iterations, numtaps 4095

### 5.1 Quality

| config | PESQ | WSR | vs stock |
|---|---|---|---|
| stock | **4.23** | **−16.7 dB** | — |
| exp2b | 3.75 | −8.3 dB | +8.4 dB louder |
| exp1b | 3.22 | −6.2 dB | +10.5 dB louder |
| exp2a | 2.91 | −9.4 dB | +7.3 dB louder |
| exp1a | 2.34 | −6.0 dB | +10.7 dB louder |

`WSR = 20·log10(rms(watermarked − original) / rms(original))` — the distortion
energy `tolerance_db` is supposed to control.

For scale, from `PROJECT_OVERVIEW.md`: AWARE solo scores PESQ 4.30, and
stacking **all three** watermark systems into one file drops it to 3.1–3.3.
exp1a at 2.34 is therefore *more audible than three stacked watermarks*.

### 5.2 Robustness — detected / total, mean bit accuracy

| attack | stock | exp1a | exp1b | exp2a | exp2b |
|---|---|---|---|---|---|
| clean | 3/3 1.00 | 3/3 1.00 | 2/3 0.98 | 3/3 1.00 | **1/3 0.83** |
| dynamic_compression | 6/6 1.00 | 6/6 1.00 | 2/6 0.93 | 6/6 1.00 | 1/6 0.78 |
| echo | 15/15 1.00 | 15/15 1.00 | 6/15 0.93 | 15/15 1.00 | 0/15 0.78 |
| mp3 | 12/15 0.98 | 11/15 0.95 | 6/15 0.84 | 12/15 0.97 | 0/15 0.72 |
| quantization | 8/15 0.85 | 12/15 0.97 | 7/15 0.88 | 11/15 0.97 | 2/15 0.73 |
| lowpass | 12/15 0.93 | 15/15 0.99 | 8/15 0.94 | 13/15 0.98 | 4/15 0.78 |
| gaussian_noise | 10/15 0.96 | 15/15 1.00 | 9/15 0.97 | 15/15 1.00 | 3/15 0.81 |
| **time_stretch** | 3/15 0.94 | 2/15 0.87 | 0/15 0.66 | **12/15 0.92** | 0/15 0.61 |
| time_jitter | 6/6 1.00 | 6/6 1.00 | 4/6 0.97 | 6/6 1.00 | 1/6 0.82 |
| highpass | 3/15 0.77 | 12/15 0.97 | 8/15 0.91 | 12/15 0.95 | 3/15 0.77 |
| encodec | 0/15 0.66 | 0/15 0.57 | 0/15 0.53 | 0/15 0.55 | 0/15 0.49 |
| background_noise | 13/15 1.00 | 15/15 1.00 | 10/15 0.98 | 15/15 1.00 | 4/15 0.82 |
| opus | 15/15 1.00 | 15/15 1.00 | 8/15 0.96 | 15/15 1.00 | 3/15 0.79 |

Attack set is the METAPXYL mastering proxy plus the key VoxWatermark families,
matching `exp_v12` so the numbers are comparable. `encodec` at 0/15 for
everything is the known neural-codec ceiling from earlier work.

---

## 6. Findings

### 6.1 Critical sampling did NOT pay off — exp2a dominates exp1a

| | exp1a (critical) | exp2a (overcomplete) |
|---|---|---|
| WSR | −6.0 dB | **−9.4 dB** (3.4 dB quieter) |
| PESQ | 2.34 | **2.91** |
| time_stretch | 2/15 | **12/15** |
| everything else | tied within ±2 cells | |

exp2a is **quieter, sounds better, and is at least as robust.** This conclusion
survives the loudness confound in §7 because the confound points the *wrong
way*: exp2a is the quieter arm, so loudness would predict it losing, not
winning.

The hypothesis in §1 — that removing overcomplete directions frees the
optimiser — is not supported. The perceptual clamp appears to have been pinning
those directions already, exactly as the "it does not help" reading predicted.

**`time_stretch` 12/15 vs 2/15 is the largest single effect in the run and is
not yet explained.** Frame count does not account for it: exp2a has 163 frames,
exp1a 61, but stock has 610 and only scores 3/15. Worth its own experiment.

### 6.2 Salient gating buys quality, and it replicates

| | all cells | gated | Δ PESQ | Δ WSR |
|---|---|---|---|---|
| critical | exp1a 2.34 | exp1b **3.22** | **+0.88** | 0.2 dB (matched) |
| overcomplete | exp2a 2.91 | exp2b **3.75** | **+0.84** | gated arm is 1.1 dB *louder* |

**exp1a vs exp1b is the cleanest comparison in the whole run** — identical WSR,
so no loudness correction is needed. Same distortion energy, +0.88 PESQ purely
from placing it in masked onset regions instead of spreading it into quiet
frames.

exp2 replicates it at +0.84, and more strongly: exp2b achieves better quality
while being *louder* than exp2a.

### 6.3 …but gating is very expensive in robustness

exp1b loses detections across the board and **fails one clean clip** (conf
0.377). exp2b is worse still — **1/3 on clean audio, 0/15 on echo and mp3**.

A config that fails on *unattacked* audio is starved, not merely "less robust."
exp2b should be treated as **not yet working** rather than as a measured
trade-off.

### 6.4 Predictions that held

- **Decimation phase is not a problem.** `time_stretch` and `time_jitter` were
  the risk — a shifted decimation phase contributes a *constant* phase rotation
  across the band, which a magnitude spectrogram cannot see. exp1a tracks stock
  on both (2/15 vs 3/15; 6/6 vs 6/6). The concern was unfounded.
- **PESQ degrades with iterations.** exp1a went 2.75 → 2.34 from 60 to 400
  iterations, because `push_extremes` keeps inflating |activation| after the
  bits are already correct.

---

## 7. Known confound — read before quoting any number against `stock`

At tol 6 **every strip arm embeds 7–11 dB louder than stock.** Robustness
tracks loudness, so all four strip-vs-stock comparisons in §5.2 are
contaminated.

Cause: `tolerance_db` is applied to the **host**, and for the strip arms AWARE
peak-normalises that host — an isolated band ~18 dB quieter than the mix —
while the residual `lo` stays at original scale. So ±6 dB on the host is not
±6 dB on the output.

This is **inherited from the band_steer chain, not introduced by critical
decimation**: exp2a shows the same gap (−9.4 dB) running entirely through the
pre-existing `chain_embed.py`. `chain_embed.py` predicted it and deferred it:

> the budget is not calibrated to the original masker. Fixing that is its own
> step; `budget_gap_db` below measures how wrong it is.

`budget_gap_db` reported 18.0 dB; the measured WSR gap is 10.7 dB, so
`budget_gap_db` **overstates** the effect by ~7 dB and is a loose predictor.

**Unaffected by this confound:** the §6.2 gating results (arms match each other
in WSR) and the §6.1 sampling result (the confound disfavours the winner).

---

## 8. Open items

1. **Fix the budget scaling** — rescale the watermarked host back to the raw
   host's level before synthesis, in both `chain_embed_critical.py` and
   `chain_embed.py`. Then §5.2's stock comparisons become meaningful.
2. **Diagnose exp2b** — 1/3 on clean audio is a fault, not a trade-off. Check
   its coverage (the `coverage` column in the CSV) and whether the 4/15 anchor
   mapping is landing where intended.
3. **Explain `time_stretch` 12/15 vs 2/15** — the largest unexplained effect.
4. **Clip 2 is an outlier** — every config is loudest there (stock −11.5 dB vs
   −19 on the others) and it is the clip exp1b failed. Possibly a low-energy
   strip driving the normalisation scale-up harder.
5. **Pre-existing bug in `cascade/vox_attacks.py`** — `VOX_GRID['opus']` builds
   `b*16` for `b=31` → 496 kbps, above libopus's 256 kbps ceiling. That
   condition silently drops (15/18 instead of 18/18) here and in every earlier
   experiment using that grid, plus it floods the logs with ffmpeg errors.

---

## 9. Reproduce

```bash
conda activate wmcompare
sbatch fsss/exp_v16_full.sbatch
```

Or step by step:

```bash
python -m fsss.salient_mapped                                       # mask arithmetic, seconds
python -m fsss.band_critical                                        # DSP T1-T6, seconds
python -m fsss.exp_v16_critical_decimation --clean-only --numtaps 4095
python -m fsss.exp_v16_critical_decimation --numtaps 4095
python -m fsss.viz_critical
```

Useful flags: `--only stock,exp1a,exp2a` trims the config set, `--iters 60`
shakes out plumbing cheaply, `--tols 6,9,12` sweeps tolerance (one table per
tolerance plus a PESQ/WSR ladder), `--strip N` picks a different strip.

Outputs land in `fsss_out/`: `exp_v16_critical_decimation.csv` and
`v16_f1_folds.png` … `v16_f6_curves.png`.

### Figures

| figure | shows |
|---|---|
| F1 | why the decimation factor must equal the band count (schematic, no audio) |
| F2 | original / isolated strip / host as AWARE reads it — the relabel, visually |
| F3 | round-trip null — the chain's cost before any watermark |
| F4 | where the watermark landed, confined to the strip |
| F5 | the exp1b salient mask over the host, with achieved coverage |
| F6 | robustness vs attack strength, one line per config |
