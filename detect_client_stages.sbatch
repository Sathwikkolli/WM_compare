#!/bin/bash
#SBATCH --job-name=wm_client_detect
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ksathwik/wm_compare/client_detect_%j.log
# GPU is optional here (detection is light). For a speedup uncomment:
#   #SBATCH --partition=spgpu
#   #SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export WM_COMPARE_BASE=$HOME/wm_compare
cd "$WM_COMPARE_BASE"
echo "start $(date)"

# Sanity: confirm the three adapters load before we trust any result (fails in seconds).
python cascade/run_cascade.py --selftest || { echo "SELFTEST FAILED -- aborting"; exit 1; }

# Detect on every client stage file (auto-downloads client_processed/ if missing).
# To go one stage at a time instead, replace the line below with e.g.:
#   python detect_client_stages.py "client_processed/EP_FAMILY NOW_Stage 02.wav"
python detect_client_stages.py

echo "done $(date)"
