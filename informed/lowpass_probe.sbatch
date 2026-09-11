#!/bin/bash
# informed/lowpass_probe.sbatch -- AWARE on unwatermarked lowpassed/highpassed speech.
# 50 clips x 21 settings = 1050 detections, no embedding. Submit with:
#     sbatch informed/lowpass_probe.sbatch
#SBATCH --job-name=lpprobe
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=lpprobe_%j.log
#SBATCH --account=hafiz1

export WM_COMPARE_BASE=$HOME/wm_compare
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare
cd "$WM_COMPARE_BASE/informed"

python lowpass_probe.py
