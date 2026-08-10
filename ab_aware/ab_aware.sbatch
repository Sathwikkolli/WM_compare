#!/bin/bash
#SBATCH --job-name=ab_aware
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=00:30:00
# 11 conditions on one ~10 s clip. The slowest cells are the three codec
# round-trips (ffmpeg subprocess) and AWARE detection itself. 30 min per clip is
# heavy headroom; keep it short so tasks schedule quickly rather than sitting
# behind multi-hour reservations.
#SBATCH --array=0-39
#SBATCH --output=ab_aware_%A_%a.log
#SBATCH --account=hafiz1

# --- AWARE detection A/B: 20 watermarked + 20 clean x 11 conditions ----------
#
# Array indices 0-19  -> wm00..wm19 (positives)
#               20-39 -> cl00..cl19 (negatives)
# run_ab.py maps SLURM_ARRAY_TASK_ID onto that order itself.
#
# CPU partition on purpose: AWARE detection runs fine on CPU for ~10 s clips and
# `standard` queues far sooner than `gpu`. Add --partition=gpu --gres=gpu:1 if
# you want it on a GPU; --account is required either way.
#
# Each task writes results/<run>/data/raw_<clip_id>.csv, so a failed task can be
# requeued alone:  sbatch --array=7 ab_aware.sbatch

export WM_COMPARE_BASE=$HOME/wm_compare
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}
export AB_RUN=${AB_RUN:-2026-08-10_aware-detection-ab}
export AWARE_MODEL=${AWARE_MODEL:-AWARE}

export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare

# module load ffmpeg           # uncomment if ffmpeg is not already on PATH
                               # (needed for mp3 / aac / opus -- 3 of 10 attacks)

cd "$WM_COMPARE_BASE/ab_aware"

# Both of these are built on the login node BEFORE submitting. Doing either here
# would race 40 tasks against the same files, and one bad path would fail all 40
# instead of failing once, interactively, in seconds.
if [ ! -f clips.json ]; then
    echo "ERROR: clips.json missing. Run 'python make_clips.py' first." >&2
    exit 1
fi
if [ ! -f work/payloads.json ]; then
    echo "ERROR: work/payloads.json missing. Run 'python embed.py' first." >&2
    exit 1
fi

python run_ab.py

# --- after the whole array finishes, run once on a login node: ----------------
#   conda activate wmcompare && cd $WM_COMPARE_BASE/ab_aware
#   python analyze.py          # -> summary.md + data/metrics.csv + figures/
# -----------------------------------------------------------------------------
