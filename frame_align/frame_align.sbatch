#!/bin/bash
#SBATCH --job-name=framealign
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
# PHASE 1 (native references only) is small: 120 draws x 2 experiments x 2
# methods = 480 aligner calls per clip, on references of 9-28 s. The bake-off
# measured gcc_phat at 0.061 s and aof at 0.109 s per whole-clip pair, so this
# should land near 1-2 minutes. 30 min is deliberate headroom for aof, whose
# cost against a short frame has never been measured.
#
# PHASE 2 IS A DIFFERENT JOB. 180 s references are ~20x the correlation length
# and the null partners must be rebuilt per partner clip. Re-time from phase 1's
# logs and raise --time before adding --ref-kinds native,60,180; do not just
# submit this file with more kinds and hope.
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

# PHASE 1. Drop --ref-kinds once phase 1 has been read and re-timed.
python run_frames.py --ref-kinds native

# --- after the whole array finishes, run once on a login node: ----------------
#   conda activate wmcompare && cd $WM_COMPARE_BASE/frame_align
#   python score_frames.py
# Then write the conclusion into results/2026-08-18_frame-align-null/README.md
# BY HAND -- results/README.md rule 2.
# -----------------------------------------------------------------------------
