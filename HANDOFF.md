# Deployment handoff — 3-watermark audio system

This repo benchmarks and integrates **three third-party audio watermarking
models** and embeds a **per-user ID** so watermarked audio can be traced back to
a user. This document is what the deployment team needs to run it on a server.

> **These are not custom models.** They are public research systems (Meta's
> AudioSeal, deepmarkpy's AWARE, the TimbreWatermarking paper's model). Our
> contribution is the unified `embed`/`detect` layer + per-user keying.

## The two functions you call

Everything is behind `cascade/service.py`:

```python
import service
service.warmup()                              # load all 3 models once at startup

# Embed the same user id (0..1023) into any subset of models, cascaded into one file
service.embed("in.wav", "out.wav", user_id=42)                       # all three
service.embed("in.wav", "out.wav", 42, models=("audioseal", "timbre"))  # a subset

# Detect: recover the user id from every watermark present, with a consensus
service.detect("out.wav")
# -> {'consensus_user_id': 42, 'agreement': '3/3', 'per_model': {...}}
```

CLI equivalent:
```bash
python service.py embed  in.wav out.wav 42
python service.py detect out.wav
```

## User IDs (uniqueness)

* Valid range **0 .. 1023** (1024 users). The ceiling is Timbre's 10-bit payload;
  the same id is embedded in all three models for redundancy.
* **Assign ids sequentially from your database** (0, 1, 2, …) to guarantee
  uniqueness. `user_key.username_to_id()` exists for convenience but **hashes into
  1024 slots, so collisions start after ~40 users** — do not use it for real
  uniqueness.
* Keep a `user_id -> user` table on your side. The watermark only carries the
  short id; identity lookup is your database's job.

## What you must obtain (not all in this git repo)

| Model | How to get it | Weights |
|---|---|---|
| **AudioSeal** | `pip install audioseal==0.2.0` | auto-downloads (~91 MB) on first use — **server needs internet on first run**, then cached in `~/.cache/audioseal` |
| **AWARE** | clone `github.com/deepmarkpy/aware` @ `fea9c49` (tag v1.0.0); put `aware/src` on `PYTHONPATH` | **none** — optimisation-based, no checkpoint |
| **Timbre** | clone `github.com/TimbreWatermarking/TimbreWatermarking` @ `c41e7d7` | needs the pretrained checkpoint `results/ckpt/pth/compressed_none-conv2_ep_20_2023-01-17_23_01_01.pth.tar` (**33 MB, handed over separately**) + `results/wmpool.txt` |

Expected layout (or point `WM_COMPARE_BASE` at wherever you put it):
```
$WM_COMPARE_BASE/
  aware/src/...
  TimbreWatermarking/watermarking_model/{config,results/ckpt/pth/*.pth.tar,results/wmpool.txt}
  cascade/{service.py,user_key.py,cascade_lib.py}
```

## Setup

```bash
export WM_COMPARE_BASE=/path/to/wm_compare     # defaults to ~/wm_compare
conda env create -f environment.yml            # env name: wmcompare  (must be committed from cluster)
conda activate wmcompare
pip install audioseal==0.2.0
# ensure aware/src and the Timbre repo are importable (PYTHONPATH / cwd handled by adapters)
python cascade/service.py detect some_watermarked.wav   # smoke test
```

## API contract / gotchas

* **Sample rate:** everything runs at **22.05 kHz** internally; adapters resample
  to each model's native rate (AudioSeal/AWARE 16 kHz). Accept any upload; output
  is 22.05 kHz PCM16 wav.
* **Payload sizes:** AudioSeal 16-bit, AWARE 20-bit, Timbre 10-bit. Detection
  threshold used in the benchmark: `bit_acc >= 0.8`.
* **Load once, keep warm.** First call loads weights and is slow; cache the
  adapters (`service.warmup()` at startup). See **Performance** below.
* **Robustness:** see `cascade_out/report.html` and `vox_out/vox_report.html` for
  how each watermark survives 23 attacks + the VoxWatermark sweep. All three are
  strong except under aggressive pitch-shift.
* **Redundancy:** because the same id is in all three, if any one detector reads
  cleanly you recover the user. `detect()` returns a per-model id + a consensus.

## Performance — AWARE embedding is the bottleneck (by design)

AudioSeal and Timbre embed with a **single forward pass** through a trained
encoder — milliseconds per file. **AWARE has no trained encoder.** Per its paper
(Algorithm 1), it embeds by running an **adversarial optimization loop for every
file**: `num_iterations` (default **400**) gradient steps, each a full detector
forward+backward in the STFT domain. So one AWARE embed is seconds–minutes,
dominating the wall-clock of any "all three" request.

| Model | Embed method | Speed |
|---|---|---|
| AudioSeal | one forward pass (trained encoder) | ~instant |
| Timbre | one forward pass (trained encoder) | ~instant |
| **AWARE** | **400-iter per-file optimization** | **slow (the bottleneck)** |

Detection is cheap for all three (roughly one forward pass) — the cost is
concentrated in AWARE **embedding**. Plan for it:

* **Do embedding asynchronously** (job queue + progress), never a blocking
  request — an AWARE embed can take many seconds on CPU.
* **Use a GPU** for the embed service if available; that optimization loop is
  exactly what GPUs accelerate (large speedup vs CPU).
* **`num_iterations` is a tunable knob** (AWARE config in `aware/src/aware/cards/`):
  fewer iterations = faster but weaker/lower-quality mark; more = slower, stronger.
  Benchmark a value that meets your latency budget.
* If embed latency is unacceptable and you don't need AWARE's specific robustness,
  a subset (`models=("audioseal","timbre")`) embeds near-instantly and still gives
  two independent watermarks carrying the same user id.

## Dependencies (Python 3.12.10 verified)

Validated on a clean **Python 3.12.10** env with **numpy 1.26.x** (pin `numpy<2`;
the old model code is not verified against numpy 2). `environment.yml` is the
Python 3.10 export; on 3.12 build a fresh env and install:

```bash
conda create -n wmcompare312 -c conda-forge python=3.12.10 -y   # 3.12.10 is on conda-forge, not defaults
conda activate wmcompare312
pip install "numpy<2" scipy soundfile librosa pyyaml pydub julius transformers audioseal
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ./aware        # AWARE's transitive deps (webrtcvad, resampy, ...) — easy to miss
```
Verify `python -c "import numpy; print(numpy.__version__)"` shows **1.26.x**, then
run the smoke test. (`webrtcvad` needs a C compiler; use `webrtcvad-wheels` if it
won't build.)

## Files that make up the deliverable

| File | Role |
|---|---|
| `cascade/service.py` | the `embed`/`detect` API you call |
| `cascade/user_key.py` | user_id <-> per-model bit messages |
| `cascade/cascade_lib.py` | model adapters, resampling, attacks, metrics |
| `environment.yml` | pinned dependencies (commit from the cluster) |
| `cascade/README.md`, `README_vox.md` | benchmark methodology + results |
