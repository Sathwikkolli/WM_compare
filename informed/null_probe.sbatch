#!/bin/bash
# informed/null_probe.sbatch -- both detectors, clean vs watermarked, lowpass + highpass.
# 250 clips x 21 settings = 5250 rows; each row = 1 AWARE detect + 2 informed scores.
# Plus 50 embeddings (~15 s each). Submit with:
#     sbatch informed/null_probe.sbatch
#SBATCH --job-name=nullprobe
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=06:00:00
#SBATCH --output=nullprobe_%j.log
#SBATCH --account=hafiz1

export WM_COMPARE_BASE=$HOME/wm_compare
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare
cd "$WM_COMPARE_BASE/informed"

python null_probe.py
