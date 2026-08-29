#!/bin/bash
#SBATCH --job-name=musicsweep
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=00:20:00
# TIMING -- MEASURED, not estimated. One clip (embed + 14 SNRs x 2 arms, each
# with an AWARE detect, a DNSMOS and a PESQ) took 44.5 s wall on a login node.
# 20 min is ~27x headroom for a single-clip task; keeping it tight matters more
# than the margin, because short jobs are scheduled far sooner than long ones.
#
#SBATCH --array=0-49
#SBATCH --output=musicsweep_%A_%a.log
#SBATCH --account=hafiz1

# --- Phase A-1: locate the quality/detection crossing -------------------------
#
# We know AWARE's confidence crosses 0.50 near +4.9 dB speech-to-music SNR.
# We do not know where the audio stops being usable. This measures both, on 50
# clips instead of the single clip the +4.9 dB figure rests on.
#
# CPU partition on purpose: AWARE detection and DNSMOS both run fine on CPU for
# 10 s clips, and `standard` queues far sooner than `gpu`.
#
# One array task per clip -> results/<run>/data/music_clip<NNN>.csv.
# Re-running a task overwrites only that clip's file, so failures requeue
# individually.

export WM_COMPARE_BASE=$HOME/wm_compare

export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}

export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare          # + requirements_extra.txt (speechmos, pesq, pystoi)

cd "$WM_COMPARE_BASE/informed"

# clips.json must exist BEFORE submitting. Building it here would race 50 tasks
# against one file, and a bad EMILIA_CSV would fail all 50 instead of failing
# once, interactively, in two seconds.
#     python clips.py
if [ ! -f clips.json ]; then
    echo "ERROR: clips.json missing. Run 'python clips.py' first." >&2
    exit 1
fi

# THE QUALITY BACKEND GATE.
#
# The entire point of this sweep is the quality axis. A run with no working
# no-reference scorer produces detection numbers we already have and nothing
# new. quality.py checks that the backend loads AND that its score falls
# monotonically as noise is added -- a backend that dips and recovers cannot
# locate a crossing point, which is the one thing we need it for.
#
# This is not ceremony: the SQUIM fallback failed exactly this check on both
# signals it could be tested against locally.
if ! python quality.py; then
    echo "ERROR: quality backend failed its self-test. Refusing to run." >&2
    echo "       pip install -r requirements_extra.txt" >&2
    exit 1
fi

python music_sweep.py

# --- after the array finishes, run once on a login node: ----------------------
#   conda activate wmcompare && cd $WM_COMPARE_BASE/informed
#   python score_music_sweep.py && python plots_music_sweep.py
#
# Then write the conclusion into
# results/2026-08-28_informed-detection/PHASE_A_PLAN.md BY HAND
# (results/README.md rule 2), and update the INDEX row.
#
# The diversity probe is a SEPARATE, smaller submission -- add three contrasting
# tracks to MUSIC_SOURCES in music_sweep.py first, then:
#   python music_sweep.py --clips 0-9 --music all
# -----------------------------------------------------------------------------
