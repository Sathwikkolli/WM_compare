#!/bin/bash
#SBATCH --job-name=alignbench
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=04:00:00
#SBATCH --array=0-29
#SBATCH --output=alignbench_%A_%a.log
#SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT

# --- Alignment method bake-off: 6 methods x 20 attacks x 30 Emilia clips ------
#
# CPU partition on purpose. Every aligner is CPU-only, and `standard` queues far
# faster than `gpu`. The only GPU-adjacent piece is the encodec attack, which
# runs fine on CPU for ~10 s clips (it is the slowest attack, not a blocker).
#
# If you would rather push encodec onto a GPU, swap the header for:
#     #SBATCH --partition=gpu
#     #SBATCH --gres=gpu:1
# ...and keep --account set; the gpu partition requires it.
#
# One array task per clip. Each writes results/<run>/data/raw_clip<NNN>.csv.
# Re-running a task overwrites only that clip's file, so failed tasks can be
# requeued individually.

export WM_COMPARE_BASE=$HOME/wm_compare

# Emilia manifest -- override if the live path differs.
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}

export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare       # same env as the watermark benchmarks.
                               # audalign + audio-offset-finder are installed on
                               # top of it -- see requirements_extra.txt for the
                               # snapshot/rollback procedure.

# module load ffmpeg           # uncomment if ffmpeg is not already on PATH
                               # (needed for mp3 / aac / opus -- 13 configs)

cd "$WM_COMPARE_BASE/align_bench"

# clips.json is shared by all tasks; task 0 builds it, the rest wait for it.
if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
    python make_clips.py
else
    for i in $(seq 1 60); do
        [ -f clips.json ] && break
        sleep 10
    done
fi

python run_bench.py

# --- after the whole array finishes, run once on a login node: ----------------
#   conda activate alignbench && cd $WM_COMPARE_BASE/align_bench
#   python score.py && python plots.py
# -----------------------------------------------------------------------------
