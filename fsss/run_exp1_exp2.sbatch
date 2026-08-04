#!/bin/bash
#SBATCH --job-name=exp1_exp2
#SBATCH --account=hafiz1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/home/ksathwik/wm_compare/exp1_exp2_%j.log
#
# Runs both experiments back to back.
#
#   Experiment 1  fsss/exp1_critical.py    aware vs exp1a vs exp1b
#   Experiment 2  fsss/exp2_bandsteer.py   aware vs exp2a vs exp2b
#
# 3 speech clips, 10 s each, tolerance_db = 4 for every config in both.
#
# COST
# AWARE has no trained encoder -- it watermarks by running a 400-step
# optimisation on every file. That is the entire wall clock. 3 configs x 3 clips
# = 9 watermarks per experiment, 18 in total. The attack grid afterwards only
# re-reads existing files, so it is cheap on the GPU, but mp3/aac/opus shell out
# to ffmpeg and encodec runs its own model -- which is why 8 CPUs are requested
# alongside the GPU.
#
# WHAT EACH RUN PRINTS
#   a chain check per config   (must pass before the numbers mean anything)
#   TABLE 1  BER and confidence, per attack
#   TABLE 2  detection rate, per attack
#   TABLE 3  PESQ and SNR on clean audio
#   4 figures + a CSV in fsss_out/
#
# The two scripts are deliberately near-identical, so
#     diff fsss/exp1_critical.py fsss/exp2_bandsteer.py
# shows exactly what separates the experiments.

eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/ksathwik/wm_compare
echo "start $(date)"
python -c "import torch; print('cuda:', torch.cuda.is_available())"

echo ""
echo "############################################################"
echo "###  EXPERIMENT 1 -- critically-sampled subband"
echo "############################################################"
python -m fsss.exp1_critical

echo ""
echo "############################################################"
echo "###  EXPERIMENT 2 -- band-steered subband"
echo "############################################################"
python -m fsss.exp2_bandsteer

echo ""
echo "outputs in fsss_out/:"
ls -la fsss_out/exp1_* fsss_out/exp2_* 2>/dev/null

echo "done $(date)"
