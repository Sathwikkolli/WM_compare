# mpeg_wm — perceptual-masking watermark (Swanson et al.) vs neural watermarks

Implements Swanson, Zhu, Tewfik & Boney, *Robust audio watermarking using
perceptual masking* (Signal Processing 66, 1998) as a **blind, multibit**
scheme. It then benchmarks it against AudioSeal, AWARE and Timbre on 120 bona fide
ASVspoof2019 LA clips under the VoxWatermark attack suite.

## The scheme

| Step | Where | What |
|---|---|---|
| frequency mask | `psychoacoustic.py` | MPEG-1 psychoacoustic model 1 per 512-sample block (32 ms at 16 kHz): dB SPL spectrum (full-scale sine = 96 dB), tonal and noise maskers, threshold in quiet, decimation, spreading function, global threshold |
| temporal mask | `psychoacoustic.py` | peak-follower envelope: 50 ms post-masking decay forward in time, 2 ms pre-masking backward |
| embed | `embed.py` | per block: PN from (key, block) → shaped to the block's mask → faded by the envelope → x(±1) for bit `b mod K` → Hann overlap-add (hop 256) |
| detect | `detect.py` | blind: re-estimate the mask from the received audio, whiten the block spectrum by it, correlate with the block's PN, sum per bit; sign = bit, `t_i = z_i/sqrt(Σr²)`, `Σt² ~ χ²(K)` → p-value |

Payload is 16 bits, repeated across the clip (about 19 blocks per bit in 5 s).

**Departures from the paper.** These are our choices, not the paper's:
- *Multibit, blind.* The paper detects one watermark and uses the original
  audio. Here each block carries a sign for one payload bit, and the detector
  never sees the original.
- *Sample rate.* The MPEG Δ-neighbourhood table is expressed in Hz, so it
  stays correct at 16 kHz.
- *Temporal masking constants.* The 50 ms / 2 ms values and the peak-follower
  envelope are our choices, not taken from the paper.
- *No sync search.* Anything that shifts the block grid (time stretch, jitter,
  positive phase shift) is expected to hurt.
- *Inverse polarity* flips every bit by design.

## Files

| File | Role |
|---|---|
| `psychoacoustic.py`, `embed.py`, `detect.py` | the scheme |
| `test_mpeg.py` | sanity tests (calibration, maskers, round trip, null, noise), no data needed |
| `make_clips.py` | 120 bona fide ASVspoof2019 LA eval clips, speakers round-robin → `clips120.json` (login node) |
| `bench_run.py` | one clip: 4 systems × {ship, match@PESQ 4.2, null} × all Vox configs → `results/<run>/<clip>/rows.csv` |
| `bench.sbatch` | Slurm array 0–119 |
| `bench_aggregate.py` | `summary.md`, `by_config.csv`, `by_family.csv`, heatmaps, bar charts |

The baseline adapters (AudioSeal, AWARE, Timbre), PESQ matching, binomial
payload test and attack runner are reused from `casp/bm_core.py`. The attacks
themselves come from `cascade/vox_attacks.py`.

## Run on Great Lakes

```bash
cd $WM_COMPARE_BASE && git fetch && git checkout mpeg-vox-bench && git pull
conda activate wmcompare && cd mpeg_wm
python test_mpeg.py                        # ~10 s, no models
python make_clips.py                       # login node, once
srun --account=hafiz1 --partition=standard --cpus-per-task=8 --mem=32g --time=00:45:00 \
     python bench_run.py --index 0 --selftest
sbatch bench.sbatch
python bench_aggregate.py                  # after the array finishes
```

Results go to `results/2026-10-19_mpeg-vox-bench/` (override with
`MPEG_BENCH_RUN`). Listening wavs (host plus every system's ship and match
versions) go to `mpeg_wm/work/<run>/<clip>/`, which is gitignored.

## Operating points

- **ship**: every system at its default strength. MPEG puts the watermark at
  the masking threshold (`--strength-db 0`).
- **match**: every system's residual is rescaled to PESQ-wb 4.2 on that clip,
  which is the fair robustness comparison.
- **null**: the unwatermarked clip through the same attacks and detectors. It
  gives the empirical false-alarm rate of each system's payload test.
