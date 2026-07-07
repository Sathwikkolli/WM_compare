#!/bin/bash
#SBATCH --job-name=aura_battery
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=aura_battery_%j.log

# ── EDIT THESE ────────────────────────────────────────────────────────────────
CKPT="$HOME/projects/aura_watermark/checkpoints/run_002/step_0200000_final.pt"
INPUT="../audio/client_original.mp3"        # <-- your source audio
OUTDIR="aura_battery_out"
# ──────────────────────────────────────────────────────────────────────────────

module load ffmpeg 2>/dev/null || true      # or: conda install -c conda-forge ffmpeg
source activate wmcompare312 2>/dev/null || conda activate wmcompare312

cd "$SLURM_SUBMIT_DIR" || exit 1

python aura_battery.py \
    --checkpoint "$CKPT" \
    --input      "$INPUT" \
    --outdir     "$OUTDIR"

echo "Battery complete -> $OUTDIR/results.md"
