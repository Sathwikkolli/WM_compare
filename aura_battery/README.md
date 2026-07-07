# AURA watermark robustness battery

Embeds an AURA 32-bit watermark into an audio file and runs the standard 8-case
robustness test battery (the same criteria as the reference sheet), producing a
results folder with every attacked file + a scorecard.

## Run on Great Lakes

```bash
cd WM_compare/aura_battery

# option A: interactive (grab a GPU first, e.g. salloc ... --gres=gpu:1)
python aura_battery.py \
    --checkpoint $HOME/projects/aura_watermark/checkpoints/run_002/step_0200000_final.pt \
    --input      ../audio/client_original.mp3 \
    --outdir     aura_battery_out

# option B: batch
sbatch run_battery.sbatch      # edit CKPT / INPUT at the top first
```

## What it produces (`aura_battery_out/`)

| File | Test Criteria |
|------|---------------|
| `_watermarked_master.wav` | full watermarked audio (source of all attacks) |
| `test1_quality.wav` | Quality — no attack |
| `test2_mp3_64k.mp3` / `_128k` / `_320k` | Compression |
| `test3_format_chain.wav` | Format chain (wav→mp3→wav→flac→wav) |
| `test4_editing.wav` | Editing (trim + insert silence + cut slice) |
| `test5_signal.wav` | Signal processing (EQ + pitch shift + normalize) |
| `test6_platform_opus.wav` | Platform simulation (Opus 24 kbps) |
| `test7_rerecord_sim.wav` | Re-recording sim (reverb + band-limit + noise) |
| `results.csv` / `results.json` / `results.md` | scorecard |

## How detection works

AURA is a **fixed 2 s / 48 kHz / 32-bit** model with no synchronization layer.
The battery tiles the watermark across the whole file in non-overlapping 2 s
windows, then decodes each window and reports:

- **Detection Probability** = mean per-window bit accuracy (1 − BER)
- **Detected** when that is ≥ `DETECT_THRESHOLD` (0.70, editable in the script)
- **Decoded bits** = per-bit majority vote across windows

## Expected weak spot

`test5_signal.wav` applies a **pitch shift**, which changes the time base and
desyncs the 2 s windows — AURA has no sync mechanism, so expect this case to
fail (accuracy near chance). This matches the reference system, whose
"Signal processing" case was also *Not Detected*. Fixing it needs either a
sync/search step at detection time or retraining with pitch robustness (the
200k model's `pitch`/`speed_pitch` BER was already ~0.4).

## Requirements

- `wmcompare312` conda env (torch, torchaudio) — or the aura training env
- `ffmpeg` on PATH (`module load ffmpeg` or `conda install -c conda-forge ffmpeg`)
- The `aura_watermark` repo checked out (script auto-finds it at
  `../../aura_watermark` or `~/projects/aura_watermark`)
