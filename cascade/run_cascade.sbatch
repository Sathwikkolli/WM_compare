#!/bin/bash
#SBATCH --job-name=wm_cascade
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=/home/ksathwik/wm_compare/cascade_%j.log
# NOTE: AWARE embedding is a 400-iteration optimization that runs once per config,
# so the full 15-config sweep is heavy on CPU (the 4h run above only reached 11/15).
# The script now writes CSVs incrementally and RESUMES, so re-running just finishes
# the remaining configs. For a ~10x speedup use the GPU partition you already use
# for the aura job: set the two lines below and the code auto-uses CUDA.
#   #SBATCH --partition=spgpu
#   #SBATCH --gres=gpu:1

eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

CASC=/home/ksathwik/wm_compare/cascade
cd "$CASC"
echo "start $(date)"

# 1) sanity-check the three adapters first (fails fast in seconds if an API is off)
python run_cascade.py --selftest || { echo "SELFTEST FAILED -- aborting"; exit 1; }

# 2) full run: singles + pairs + triples (15 configs)
python run_cascade.py --sizes 1,2,3

# 3) build figures + HTML report
python make_report.py

echo "done $(date)"
