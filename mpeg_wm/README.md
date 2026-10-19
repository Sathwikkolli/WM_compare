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

## CASP placement: CASP decides where, MPEG decides how much

`casp_model.py` is the revised CASP model (Paulick, Relaño-Iborra & Dau, JASA
157:3232, 2025), written from the paper and the papers it cites. It runs one
function per stage:

| Stage | Function |
|---|---|
| Pascals, full scale = 96 dB SPL | `to_pascals` |
| 53 channels, 250 Hz–8 kHz, 2 per ERB | `erb_channels` |
| outer/middle ear (Lopez-Poveda & Meddis 2001) | `outer_middle_ear` |
| DRNL cochlea | `gammatone`, `drnl_params`, `broken_stick`, `drnl` |
| nonlinear inner hair cell | `met_conductance`, `inner_hair_cell` |
| adaptation loops, per-channel floors from ISO 389-7 | `adaptation_min_levels`, `adaptation` |
| modulation filterbank | `modulation_filterbank` |
| whole model | `internal_representation` |
| decision stage | `ir_distance`, `dprime` |
| calibration (1 kHz, 60 dB, 1 dB step = d′ 1) | `calibration_distance`, `check_calibration` |

**Where it differs from the vendored `casp/vendor/torch_amt` port.** We follow
the paper in each case:
- the channel grid;
- the ear filter;
- the DRNL linear-path exponents;
- holding the compressor constant above 1.5 kHz as `min(CF, 1500)`, where the
  port jumps at 1 kHz;
- the order of the two terms in the Boltzmann function (Eq. 1);
- the modulation limit, which is 1000 Hz rather than 150 Hz.

The potassium reversal potentials are printed without a sign in Table I; we use
−71 and −78 mV.

`casp_place.py` keeps the MPEG watermark's total energy for the clip and lets
CASP move it between tiles. Tiles are 12 ERB bands across 250 Hz–8 kHz × 128 ms.
Each round:
1. synthesise the watermark;
2. run CASP once on host + watermark, giving a d′ per tile;
3. turn tiles towards equal audibility;
4. rescale to MPEG's energy.

The two ways and their controls:

| System | Starting point | Rounds |
|---|---|---|
| CASP-1 | the MPEG watermark (mask + envelope) | 4, step 1 |
| CASP-2 | flat noise, 250 Hz–8 kHz | 6, step 3 |
| CASP-1-shuf / CASP-2-shuf | partner's final tile gains, permuted across tiles | — |

The shuffled versions keep how uneven the map is but destroy where CASP put the
energy. The payload layout, key and blind detector are MPEG's (`detect.decode`).

```bash
python test_casp.py                                        # model + placement tests
sbatch --array=0-119%30 bench_casp.sbatch                  # results/2026-10-20_casp-vox-bench/
MPEG_BENCH_RUN=2026-10-20_casp-vox-bench,2026-10-19_mpeg-vox-bench \
MPEG_BENCH_OUT=2026-10-20_casp-vox-bench python bench_aggregate.py
```

The merged `summary.md` includes paired, clip-by-clip comparisons: CASP-x vs
MPEG, CASP-x vs its shuffled control, and CASP-1 vs CASP-2. Each comparison
reports the detection rate and PESQ with bootstrap intervals. At `ship` all
CASP systems carry exactly MPEG's energy, so any difference there is placement.

## Operating points

- **ship**: every system at its default strength. MPEG puts the watermark at
  the masking threshold (`--strength-db 0`).
- **match**: every system's residual is rescaled to PESQ-wb 4.2 on that clip,
  which is the fair robustness comparison.
- **null**: the unwatermarked clip through the same attacks and detectors. It
  gives the empirical false-alarm rate of each system's payload test.
