#!/bin/bash
#SBATCH --job-name=emilia_wm4
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=08:00:00
#SBATCH --output=emilia_wm4_%j.log
##SBATCH --account=REPLACE_WITH_YOUR_ACCOUNT   # <-- uncomment + set (gpu partition needs it)

# --- Great Lakes GPU run: 4 watermarks x 50 Emilia clips x 3 messages ----------
export WM_COMPARE_BASE=$HOME/wm_compare
export AURA_CKPT=$HOME/projects/aura_watermark/checkpoints/run_002/step_0200000_final.pt
# AURA's speaker-disjoint held-out Emilia val split (no train leakage for AURA):
export EMILIA_CSV=$HOME/projects/aura_watermark/data/val.csv
export TORCHDYNAMO_DISABLE=1

# activate the env that has all 4 models (verified: audioseal/aware/timbre/aura load)
eval "$(conda shell.bash hook)"
conda activate wmcompare
# module load ffmpeg          # only if ffmpeg is not already on PATH (mp3/aac/opus attacks)
cd "$WM_COMPARE_BASE/cascade"

# All stages are resumable; if the job is requeued it continues where it stopped.
python emilia_bench.py select
python emilia_bench.py embed
python emilia_bench.py attack
python emilia_bench.py fpr
python emilia_bench.py aggregate
