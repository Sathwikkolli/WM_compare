# ab_aware — AWARE detection A/B (watermarked vs. unwatermarked)

Answers the question every robustness table in this repo leaves open: when AWARE
says "watermarked", is it detecting the **watermark**, or just responding to
speech? Every prior benchmark here scores only watermarked audio, so none of
them carries a false-positive rate.

Results land in `results/2026-08-10_aware-detection-ab/` per the archive
convention. **Read that README first** — the design, the pre-registered
prediction, and the limits live there.

## Run it

```bash
conda activate wmcompare
cd $WM_COMPARE_BASE/ab_aware

# sanity checks before burning cluster time
python attacks_ab.py            # prints the 11 conditions; probes availability

# login node -- NOT in the array (40 tasks would race these files)
python make_clips.py            # -> clips.json     (20 wm + 20 clean)
python embed.py                 # -> work/          (embeds + round-trip check)

# the real run
sbatch ab_aware.sbatch          # array 0-39, one task per clip

# after the array finishes
python analyze.py               # -> summary.md + data/metrics.csv + figures/
python plot_confusion.py        # -> figures/confusion.png        (pooled, 2 panels)
python plot_confusion.py --grid # -> figures/confusion_grid_*.png (every condition)
```

`embed.py` prints a clean-audio round-trip check and warns loudly if any
positive is undetected **before** any attack. If that fires, stop — every
downstream number would be noise.

## Files

| File | Does |
|---|---|
| `make_clips.py` | Picks 20 + 20 Emilia clips from one shuffled pool -> `clips.json` |
| `embed.py` | Embeds the 20 positives, per-clip random 20-bit payload -> `work/` |
| `attacks_ab.py` | The 10-distortion grid + `clean` control |
| `run_ab.py` | Slurm-array runner, one task per clip -> `data/raw_<id>.csv` |
| `analyze.py` | All statistics -> `summary.md`. **Metric meanings documented at the top.** |
| `plot_confusion.py` | The confusion matrix as a figure, at both thresholds. Imports `analyze.py` rather than recomputing, so it cannot disagree with `summary.md`. |
| `ab_aware.sbatch` | Great Lakes job, array 0-39 |

## What makes this benchmark trustworthy

**1. There is a negative control.** This is the entire point. 20 unwatermarked
clips go through the same 11 conditions and the same code path as the positives.
Without them a stuck detector scores 100% and looks perfect.

**2. The threshold is derived, not assumed.** `run_ab.py` writes raw confidence
and never thresholds. `analyze.py` reports the confusion matrix at both the
codebase default (`conf >= 0.5`) and a threshold calibrated on the negatives.
Baking 0.5 into the raw data would make ROC/AUC impossible after the fact.

**3. The claims are sized to the sample.** 20 negatives cannot certify a low
FPR, so no TPR@FPR=1e-3 is reported — the finest resolvable FPR is 1/20 = 5%.
Confidence intervals resample **clips**, because 11 rows from one clip are not
11 independent observations. The pooled confusion counts carry no interval at
all and say so.

**4. The prediction was written down before the run**, in the results README.

## Known gotchas

- **`time_stretch` needs librosa**, which is absent from a bare local checkout
  but present in the cluster `wmcompare` env. `python attacks_ab.py` reporting it
  as unavailable locally is expected, not a failure.
- **`aac` returns more samples than it was given** (measured +128 on speech-like
  input, 2026-08-10). It is *trailing* padding — lag measured 0 for all three
  codecs — so the `min(len(...))` truncation in `run_ab.py` handles it. Re-check
  if the ffmpeg version changes.
- **PESQ/SNR are blank for `time_stretch_1.1` and `crop_50`.** Both break sample
  alignment, so a number there would measure the misalignment rather than the
  distortion.
- **PESQ means different things per row.** Only `wm`/`clean-condition` is
  watermark-only cost. Every attacked row mixes in the attack's own damage; the
  `clean` arm's attacked rows are the control for that.
