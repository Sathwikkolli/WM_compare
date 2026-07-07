#!/bin/bash
#SBATCH --job-name=vox_aura
#SBATCH --account=hafiz1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ksathwik/wm_compare/vox_aura_%j.log
# AURA is a 68M-param transformer and tiles the clip into many 2-s windows, so a
# GPU keeps the full 17-attack sweep to minutes instead of hours.

eval "$(conda shell.bash hook)"; conda activate wmcompare312
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# point at the trained checkpoint (or export AURA_CKPT before sbatch)
export AURA_CKPT="${AURA_CKPT:-$HOME/projects/aura_watermark/checkpoints/run_002/step_0200000_final.pt}"

CASC=/home/ksathwik/wm_compare/cascade
cd "$CASC"
echo "start $(date)  ckpt=$AURA_CKPT"

# 1) verify every attack applies + AURA loads/detects
python run_vox_aura.py --selftest

# 2) full sweep: AURA x 17 attacks x strengths
python run_vox_aura.py

# 3) merge with AudioSeal/AWARE/Timbre and emit the comparison table
python compare_vox.py

echo "done $(date)"
