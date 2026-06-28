# Multi-watermark cascade benchmark

Embed **AudioSeal + AWARE + Timbre** into a single audio file *in sequence*
(cascaded), measure how much each watermark damages the others (interference),
and how the stacked file survives 23 attacks. Covers **all 15 orderings**:
3 singles + 6 ordered pairs + 6 ordered triples.

## The core idea

Watermarking = adding a tiny, inaudible perturbation `δ(message)` to the waveform.
Cascading stacks them:

```
clean ──embed A──▶ clean+δA ──embed B──▶ clean+δA+δB ──embed C──▶ clean+δA+δB+δC
```

Each embedder assumes it's writing on a *clean* page, but B and C are actually
writing over earlier watermarks — that foreign perturbation acts as noise to the
earlier watermark's detector. So the **earliest-embedded watermark usually degrades
most**, and ordering matters. This harness measures exactly that.

## What gets measured

* **Interference** — after each embed stage, every watermark embedded *so far* is
  detected on the clean (un-attacked) intermediate. Isolates embedding damage.
* **Robustness** — the final cascaded file is hit with 23 attacks; every present
  watermark is detected per attack.
* **Quality** — PESQ, SNR, SI-SNR, STOI after each stack depth (1 → 2 → 3).
* **Controls** — `clean_negative` (false-positive floor) and `resample_only`
  (cost of the 22.05k↔16k resampling the cascade does between tools).

`bit_acc ≥ 0.8` = detected.

## Sample-rate handling

Everything is canonicalised at **22.05 kHz** (Timbre's native rate). AudioSeal and
AWARE run at 16 kHz, so the adapters resample 22.05k→16k to embed/detect and back
to 22.05k afterwards. The `resample_only` control quantifies that round-trip cost.

## Attacks (23)

`baseline, mp3_128, mp3_64, aac_128, opus_64, resample_8k, lowpass_4k,
highpass_500, volume_0.5, quantize_8bit, echo, sample_suppress, noise_20db,
noise_10db, noise_5db, tempo_up_10, tempo_down_10, denoise, crop_30s, crop_10s,
crop_5s, crop_3s, pitch_up`

## Run it (on Great Lakes)

```bash
cd ~/wm_compare/cascade

# 1. ALWAYS run the self-test first — solo embed+detect each tool (~seconds).
#    Confirms the three adapter APIs are correct before the long run.
python run_cascade.py --selftest

# 2. Full run (15 configs). Or `--sizes 3` for just the 6 triples.
python run_cascade.py --sizes 1,2,3

# 3. Figures + HTML report.
python make_report.py
```

Or submit the batch job (does all three steps):

```bash
sbatch run_cascade.sbatch
```

## Outputs  (`~/wm_compare/cascade_out/`)

| file | contents |
|---|---|
| `cascade_interference.csv` | progressive (clean) detections per config/stage |
| `cascade_robustness.csv`   | attack × watermark detections per config |
| `cascade_quality.csv`      | PESQ/SNR/SI-SNR/STOI per stack depth |
| `cascade_controls.csv`     | negative + resample-only controls |
| `figs/*.png`               | interference, robustness, order-effect, quality plots |
| `report.html`              | everything assembled — open this |
| `wavs/<config>/`           | intermediate, final, and attacked wavs |

`scp` the `cascade_out/` folder back and open `report.html`.

## Files

| file | role |
|---|---|
| `cascade_lib.py`  | tool adapters (load/embed/detect), attacks, quality metrics |
| `run_cascade.py`  | orchestrator: cascade embed, progressive detect, attacks |
| `make_report.py`  | CSV → PNG heatmaps + HTML report |
| `run_cascade.sbatch` | SLURM job |

## Adapter assumptions (verified against the official repos)

| tool | rate | bits | embed call |
|---|---|---|---|
| AudioSeal | 16k | 16 | `gen(x, sample_rate=16000, message=msg, alpha=1.0)` |
| AWARE | 16k | 20 | `embed_watermark(audio1d, 16000, bits, embedder)` (400-iter optimization) |
| Timbre | 22.05k | wmpool.txt | `encoder.test_forward(wav[1,1,N], msg±1)` |

If the self-test flags a tool, the exact call is tagged `# VERIFY` in
`cascade_lib.py` — paste the error and it's a 1–2 line fix.
