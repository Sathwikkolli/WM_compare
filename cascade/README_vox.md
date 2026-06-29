# VoxWatermark no-box attack suite

Runs the **VoxWatermark** (Interspeech 2026) no-box perturbation grid — 17 attack
families at swept strengths — against AudioSeal, AWARE, Timbre **solo** and against
the **combined 3-watermark file**, producing robustness-vs-strength curves.

Attacks are transcribed from the official repo (`Splitting_Dataset/no_box_funcs.py`)
so the parameter grids match the paper. Everything runs at the **22.05 kHz master**
(same path as the cascade study) so the two sets of results are directly comparable.
Black-box attacks (Square / HSJA) are deferred to a later phase.

## Attacks & sweeps (17 families)

| Attack | Strengths |
|---|---|
| time_stretch | 0.7, 0.9, 1.1, 1.3, 1.5 |
| gaussian_noise | SNR 40, 30, 20, 10, 5 dB |
| background_noise | SNR 40, 30, 20, 10, 5 dB *(needs a noise wav, see below)* |
| opus | 16, 32, 64, 128, 256, 496 kbps |
| encodec | 1.5, 3, 6, 12, 24 kbps *(neural codec)* |
| quantization | 4, 8, 16, 32, 64 levels |
| highpass / lowpass | ratio 0.1–0.5 of sample rate |
| smooth | window 6, 10, 14, 18, 22 |
| echo | duration 0.1, 0.3, 0.5, 0.7, 0.9 |
| mp3 | 8, 16, 24, 32, 40 kbps |
| aac | 8k, 40k |
| dynamic_compression / dynamic_expansion | (−10,2), (−30,8) |
| inverse_polarity | negate |
| time_jitter | scale 0.01, 0.5 |
| phase_shift | 1, −1000 samples |

## One-time setup (extra dependencies)

```bash
conda activate wmcompare
pip install transformers julius pydub        # encodec / filters / DRC
# EnCodec model downloads automatically on first use (facebook/encodec_24khz)
```
Each attack degrades to **SKIP** if its dependency is missing — a missing package
never blocks the rest of the run. `julius` missing → highpass/lowpass fall back to a
scipy Butterworth filter.

### background_noise (optional)
Needs a real noise clip. Drop any `.wav` into `cascade/noises/`:
```bash
mkdir -p ~/wm_compare/cascade/noises
# copy/download a noise wav into it; otherwise background_noise records SKIP
```

## Run

```bash
cd ~/wm_compare/cascade
python run_vox.py --selftest     # applies every attack once (seconds) — fix any SKIP/FAIL
python run_vox.py                # full sweep (resumable)
python vox_report.py             # curves + heatmaps + vox_report.html
# or: sbatch run_vox.sbatch
```

Subsets: `--targets combined` or `--attacks encodec,time_stretch,mp3`.

## Outputs (`~/wm_compare/vox_out/`)

| file | contents |
|---|---|
| `vox_results.csv` | per target × attack × strength × watermark: bit_acc, BER, conf, detected |
| `figs/curve_<attack>.png` | robustness vs strength — solid = solo, dashed = in combined file |
| `figs/heatmap_combined.png` | condition × watermark grid for the combined file |
| `vox_report.html` | everything assembled — open this |

## Reading the curves
Each curve overlays, for one attack, every watermark's bit-accuracy as strength
increases. **Solid = the watermark alone; dashed = the same watermark inside the
combined 3-in-1 file.** If solid and dashed overlap, stacking didn't change that
watermark's robustness (the expected result from the cascade study). The dotted line
at 0.8 is the detection threshold.
