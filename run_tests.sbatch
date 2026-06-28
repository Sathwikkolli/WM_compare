#!/bin/bash
#SBATCH --job-name=wm_tests
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=/home/ksathwik/wm_compare/wm_tests_%j.log

eval "$(conda shell.bash hook)"
conda activate wmcompare
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
cd /home/ksathwik/wm_compare/Audio-Watermark-Toolkit
echo "=== started $(date) ==="
python /home/ksathwik/wm_compare/run_babar_tests.py
echo "=== done $(date) ==="
