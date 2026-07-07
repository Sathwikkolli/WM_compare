#!/bin/bash
#SBATCH --job-name=aware_fail_spec
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=aware_fail_spec_%j.log

# --- Great Lakes: set these for your checkout -------------------------------
export WM_COMPARE_BASE=$HOME/wm_compare          # <-- point at your checkout
export TORCHDYNAMO_DISABLE=1
# module load python/3.12 cuda ffmpeg   # or: conda activate wmcompare
cd "$WM_COMPARE_BASE/cascade"

# Embeds AWARE (slow) then renders spectrograms of the failure cases.
# Add  --use-wm ../audio/aware_wm.wav  to reuse a cached watermarked clip and skip embedding.
python make_fail_spectrograms.py --seconds 6
