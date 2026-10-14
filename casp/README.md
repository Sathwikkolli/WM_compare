# casp — CASP-derived masking threshold for AWARE's STFT grid

Question: can the revised CASP auditory model (Paulick, Relaño-Iborra & Dau,
JASA 157:3232, 2025) give a better "how much may I change this bin" budget than
AWARE's fixed `|Δ| ≤ η·|X|` (6 dB)? This folder only builds and inspects that
budget. No AWARE embedding happens here.

## Method (one paragraph)

AWARE's rule is one η everywhere. Here η becomes `η(band, time block)`. For
each of 8 ERB-spaced bands over 1–4 kHz and each 256 ms block (16 STFT frames),
a random-sign, AWARE-shaped change `|Δ| = η·|X|` is added inside that block
only, the host and the changed signal go through the revised CASP, and η is
solved (log-log interpolation over 3 probe gains, η = 0.06 / 0.25 / 1.0) until `‖IR(x+Δ) − IR(x)‖`
equals the target. For a template equal to the difference signal and constant
internal noise σ, `d′ = ‖ΔIR‖/σ`, so one audibility unit is calibrated from the
paper's own fitting condition: **1 kHz tone, 60 dB SPL, intensity increment
`JND_DB`** (default 1 dB, an assumption, flagged in `validate.py`). Because a
watermark perturbs all bands at once, each band gets an equal share
`target·D_JND/√n_bands`. The whole-signal runs in `run_casp.py` check that.

## Files

| File | Role |
|---|---|
| `casp_front.py` | wraps `Paulick2024`: dB SPL scaling, reduced channel grid (700–5500 Hz, 24 ch), interpolated quiet thresholds, single IR tensor |
| `validate.py` | calibration + sanity (level growth, Weber check, 16 vs 44.1 kHz) → `validation.json/png` |
| `threshold.py` | band×block probing, solve for η, STFT-grid maps, whole-signal audibility |
| `run_casp.py` | one clip end-to-end → `probe.npz`, `summary.json`, `audibility.npz`, figures |
| `plots.py` | figures |
| `select_clips.py` | picks 4 clips spread over the pause fraction (login node) |
| `casp_threshold.sbatch` | Slurm array 0–3 |
| `vendor/` | upstream `torch-amt` 0.2.0 (GPL-3.0-or-later), unmodified; **do not edit** |

## Run on Great Lakes

```bash
cd $WM_COMPARE_BASE && git pull
cd casp
python select_clips.py                 # login node, once
python run_casp.py --synth --quick     # optional smoke test
sbatch casp_threshold.sbatch
```

Results: `results/2026-10-14_casp-threshold/` — `validation/` and one folder per clip.

## Figures per clip

| File | Shows |
|---|---|
| `fig1_overview.png` | waveform, spectrogram, CASP tolerance map |
| `fig2_vs_aware.png` | CASP allowance minus AWARE (red = CASP allows more) and its histogram |
| `fig3_tolerance_vs_time.png` | tolerance per band vs time, AWARE 6 dB reference |
| `fig4_audibility.png` | whole-signal audibility per block: AWARE tolerances vs CASP targets |
| `fig5_quality_energy.png` | PESQ / audibility vs energy of the change |
| `fig6_probe_curves.png` | `D(η)` curves and how often the solver hits its range |

## Known limits

- `torch_amt` is a third-party port with no published numerical validation; `validate.py` is the check.
- 16 kHz instead of the paper's 44.1 kHz, and 24 channels instead of 50 (cost). `--fs-compare` quantifies the first.
- Adaptation uses only 0.30 s of left context per block (time constants go to 0.5 s).
- `D_JND` rests on the assumed JND; sensitivity rows are in `validation.json`.
- Random-sign stand-in, not AWARE's optimised perturbation.

## A/B run: stock AWARE vs AWARE + CASP budget

Same frozen AWARE detector and optimiser in both arms; only the per-coefficient
bounds passed to `_optimize` differ (arm B: CASP shape rescaled to arm A's
in-band energy, iso-energy). 5 clips x (clean + MP3 64/32 kbps + white noise
20/10 dB SNR + low-pass 3.5/2.5 kHz), plus unwatermarked negatives.

| File | Role |
|---|---|
| `ab_embed_detect.py` | one clip: both arms, attacks, detection, quality, spectrograms, wavs |
| `ab_aggregate.py` | summary.md, robustness table, figures |
| `ab_casp.sbatch` | Slurm array 0-4 |

```bash
cd $WM_COMPARE_BASE && git pull && cd casp
python ab_embed_detect.py --make-clips     # login node, once
sbatch ab_casp.sbatch
python ab_aggregate.py                     # after the array finishes
```

`python ab_embed_detect.py --mock` runs the plumbing with a fake AWARE (no
cluster needed). It cannot say anything about robustness.
