#!/bin/bash
#SBATCH --job-name=mask_e0e1
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --array=0-29
#SBATCH --output=mask_e0e1_%A_%a.log

# --- E0 (AWARE baseline) + E1a (budget usage) + E1b (which windows carry the mark)
#
# One array task per clip. A 3-8 s ASVspoof clip is one 400-iteration AWARE embed
# plus 12 attacks plus ~3 x 20 one-second window detections: minutes on CPU.
# `standard` queues sooner than `gpu`; for GPU add  --partition=gpu --gres=gpu:1
#
# Before submitting (login node):
#   conda activate wmcompare && cd $WM_COMPARE_BASE/masking
#   python make_clips.py                      # -> clips.json (30 clips)
#   python run_e0_e1.py --index 0             # optional smoke test, ~minutes
#   sbatch e0_e1.sbatch
# After the array finishes (login node):
#   python aggregate.py                       # -> results/<run>/summary.md + figures/
#
# A failed task can be rerun alone:  sbatch --array=7 e0_e1.sbatch

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare
command -v ffmpeg >/dev/null || module load ffmpeg     # mp3 attacks need it

cd "$WM_COMPARE_BASE/masking"
[ -f clips.json ] || { echo "clips.json missing: run 'python make_clips.py' first" >&2; exit 1; }

echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  task: $SLURM_ARRAY_TASK_ID  host: $(hostname)"
python run_e0_e1.py --index "$SLURM_ARRAY_TASK_ID"
