#!/bin/bash
#SBATCH --job-name=vox_attacks
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ksathwik/wm_compare/vox_%j.log
# Lighter than the cascade job: embedding happens once per target (4 embeds total),
# then it's just attacks + detections. EnCodec adds model inference on 5 conditions.

eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CASC=/home/ksathwik/wm_compare/cascade
cd "$CASC"
echo "start $(date)"

# 1) verify every attack applies (catches missing deps in seconds)
python run_vox.py --selftest

# 2) full sweep: 4 targets x 17 attacks x strengths
python run_vox.py

# 3) curves + heatmaps + report
python vox_report.py

echo "done $(date)"
