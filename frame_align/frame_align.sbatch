#!/bin/bash
#SBATCH --job-name=framealign
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=01:00:00
# TIMING -- re-derived from phase 1's measured numbers, not guessed.
#
# Phase 1 (native refs, 9-28 s) MEASURED: gcc_phat 0.018 s/call median (p95
# 0.060), aof 0.033 (p95 0.065), 480 calls/clip -> ~12 s of aligner time. It ran
# comfortably inside 30 min.
#
# Phase 2 adds the 60 s and 180 s kinds: 1440 calls/clip. gcc_phat pads to
# nfft=next_pow2(ref+frame), so 180 s (nfft 4M) costs ~9x native (nfft 512k);
# aof scales closer to linear, ~11x. Estimated per clip:
#     native  480 calls  ~12 s
#     60 s    480 calls  ~40 s
#     180 s   480 calls  ~130 s
#     + pool load (~27 clips) and 20 null-partner loads   ~60-90 s
#   ------------------------------------------------------------------
#     ~5-8 min/clip  ->  1 h is 8-12x headroom
#
# The 180 s figure is EXTRAPOLATED, not measured. Before submitting phase 2, time
# one clip on a login node and compare:
#     time python run_frames.py --clip 0 --ref-kinds native,60,180
# If that exceeds ~10 min, raise --time before submitting 30 of them.
#
# Memory: the null-reference cache holds up to 20 partners x 3 kinds (~310 MB)
# plus FFT workspace at nfft=4M (~64 MB/array). 16 g is ample; it is the wall
# clock that needs watching, not the memory.
#SBATCH --array=0-29
#SBATCH --output=framealign_%A_%a.log
#SBATCH --account=hafiz1

# --- Frame-level alignment: resolution limit + false-alarm floor -------------
#
# CPU partition on purpose: gcc_phat is numpy FFT, aof is MFCC cross-correlation.
# Neither touches a GPU, and `standard` queues far sooner than `gpu`.
#
# One array task per clip -> results/<run>/data/raw_clip<NNN>.csv. Re-running a
# task overwrites only that clip's file, so failed tasks requeue individually.
# Frame draws are fixed in frames.json, so a requeued task reproduces exactly
# the rows it was meant to produce.

export WM_COMPARE_BASE=$HOME/wm_compare

export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}

export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare        # same env as the bake-off; audio-offset-finder
                                # is installed on top of it, see
                                # align_bench/requirements_extra.txt

cd "$WM_COMPARE_BASE/frame_align"

# Both JSONs must exist BEFORE submitting. Building them here would race 30
# tasks against one file, and a bad EMILIA_CSV would fail all 30 instead of
# failing once, interactively, in two seconds.
#
#   cd $WM_COMPARE_BASE/align_bench && python make_clips.py   # if not already there
#   cd $WM_COMPARE_BASE/frame_align  && python make_frames.py
if [ ! -f ../align_bench/clips.json ]; then
    echo "ERROR: align_bench/clips.json missing. Run make_clips.py first." >&2
    exit 1
fi
if [ ! -f frames.json ]; then
    echo "ERROR: frames.json missing. Run 'python make_frames.py' first." >&2
    exit 1
fi

# PHASE selects the reference kinds. Phase 1 is complete (see
# results/2026-08-18_frame-align-null/README.md); phase 2 settles prediction 2,
# whether false-alarm rate rises with reference length.
#
#   PHASE=1  native only          -- done 2026-08-18
#   PHASE=2  native, 60 s, 180 s  -- read the TIMING block above first
#
# Override at submit time without editing this file:
#     sbatch --export=ALL,PHASE=2 frame_align.sbatch
PHASE=${PHASE:-2}

case "$PHASE" in
    1) KINDS="native" ;;
    2) KINDS="native,60,180" ;;
    *) echo "ERROR: PHASE must be 1 or 2, got '$PHASE'" >&2; exit 1 ;;
esac

echo "PHASE=$PHASE  ref kinds: $KINDS"

# Phase 2 overwrites phase 1's raw_clip*.csv in the SAME results directory, and
# results/README.md rule 1 forbids that -- old numbers must stay citable. Move
# phase 1's data aside before the first phase-2 submission:
#     cd $WM_COMPARE_BASE/results/2026-08-18_frame-align-null
#     mkdir -p data_phase1 && mv data/raw_clip*.csv data/metrics.csv data_phase1/
#     cp summary.md summary_phase1.md
if [ "$PHASE" = "2" ] && [ -f "$WM_COMPARE_BASE/results/2026-08-18_frame-align-null/data/raw_clip000.csv" ] \
   && [ ! -d "$WM_COMPARE_BASE/results/2026-08-18_frame-align-null/data_phase1" ]; then
    echo "ERROR: phase 1 data is still in data/ and would be overwritten." >&2
    echo "       Archive it first -- see the comment above this check." >&2
    exit 1
fi

python run_frames.py --ref-kinds "$KINDS"

# --- after the whole array finishes, run once on a login node: ----------------
#   conda activate wmcompare && cd $WM_COMPARE_BASE/frame_align
#   python score_frames.py
# Then write the conclusion into results/2026-08-18_frame-align-null/README.md
# BY HAND -- results/README.md rule 2.
# -----------------------------------------------------------------------------
