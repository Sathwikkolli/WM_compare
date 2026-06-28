#!/bin/bash
#SBATCH --job-name=wm_tests
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=/home/ksathwik/wm_compare/wm_tests_%j.log
eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
cd /home/ksathwik/wm_compare/Audio-Watermark-Toolkit
echo "start $(date)"
python /home/ksathwik/wm_compare/run_babar_tests_par.py
echo "done $(date)"
