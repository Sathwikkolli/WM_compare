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
* **Load once, keep warm.** First call loads weights and is slow. **AWARE embeds
  via a 400-iteration optimisation — it is not instant** (order of seconds per
  file). Consider an async/queue for embed requests.
* **Robustness:** see `cascade_out/report.html` and `vox_out/vox_report.html` for
  how each watermark survives 23 attacks + the VoxWatermark sweep. All three are
  strong except under aggressive pitch-shift.
* **Redundancy:** because the same id is in all three, if any one detector reads
  cleanly you recover the user. `detect()` returns a per-model id + a consensus.

## Files that make up the deliverable

| File | Role |
|---|---|
| `cascade/service.py` | the `embed`/`detect` API you call |
| `cascade/user_key.py` | user_id <-> per-model bit messages |
| `cascade/cascade_lib.py` | model adapters, resampling, attacks, metrics |
| `environment.yml` | pinned dependencies (commit from the cluster) |
| `cascade/README.md`, `README_vox.md` | benchmark methodology + results |
