#!/bin/bash
#SBATCH --job-name=df_probe
#SBATCH --account=hafiz1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ksathwik/wm_compare/deepfake_probe/df_probe_%j.log
# 6 clips x 2 AWARE embeds (aware_only + all_three) = 12 AWARE optimizations.
# GPU makes that minutes; on `standard` (CPU) budget ~4-6 h instead.
# Run setup_detectors.sh once on the login node before submitting.

eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

cd /home/ksathwik/wm_compare/deepfake_probe
echo "start $(date)"

# fail fast: watermark adapters, then detectors
python ../cascade/run_cascade.py --selftest || { echo "WM SELFTEST FAILED"; exit 1; }
python detectors.py                         || { echo "DETECTOR SMOKE TEST FAILED"; exit 1; }

python run_probe.py "$@"
echo "done $(date)"
