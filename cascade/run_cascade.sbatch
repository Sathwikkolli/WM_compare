#!/bin/bash
#SBATCH --job-name=wm_cascade
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ksathwik/wm_compare/cascade_%j.log
# NOTE: AWARE embedding is a 400-iteration optimization and runs ~11x across the
# 15 configs, so this is much heavier than the single-model benchmarks. If a GPU
# partition is available, switch --partition and add --gres=gpu:1 to speed up
# Timbre + AWARE (the code auto-uses CUDA when torch.cuda.is_available()).

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
