#!/bin/bash
#SBATCH --job-name=v16_full
#SBATCH --account=hafiz1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=05:00:00
#SBATCH --output=/home/ksathwik/wm_compare/v16full_%j.log
#
# The complete 2x2 plus reference:
#
#                     ALL cells      salient-gated
#   CRITICAL (exp1)   exp1a          exp1b
#   OVERCOMPLETE(exp2) exp2a         exp2b
#   reference         stock
#
# 5 configs x 3 clips = 15 embeds, then the attack grid (detection only, cheap
# on GPU but ffmpeg/encodec keep it CPU-bound -- hence 8 CPUs).
#
# KNOWN CONFOUND, STATED UP FRONT
# At tol 6 the strip arms embed ~8-11 dB louder than stock, because
# tolerance_db is applied to a peak-normalised host rather than to the mix.
# Measured on 3 clips: stock -16.7 dB WSR, exp2a -9.1, exp1a -6.1, exp1b -6.2.
# So ANY comparison against stock in these tables is contaminated by loudness.
#
# What is NOT contaminated, because the arms match each other in WSR:
#   exp1a vs exp1b  (-6.1 vs -6.2 dB)  -> the gating result, clean
#   exp2a vs exp2b                     -> same question, other sampling
# exp1a vs exp2a is 3 dB apart and still needs the budget fix before it means
# anything. Read the WSR column beside every PESQ before drawing a conclusion.
#
# Steps 1-2 are the gate and cost minutes. Step 2's clean table must show
# bit_acc ~1.0 for every config; if a strip arm is below stock there, its chain
# is wrong and nothing in step 3 is worth reading.

eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/ksathwik/wm_compare
echo "start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# 1) mask arithmetic, no AWARE, seconds. Confirms both salient arms build masks
#    from the same helper and that anchor_rate 1.2 does not saturate either.
echo "=== step 1: salient_mapped smoke test ==="
python -m fsss.salient_mapped

# 2) DSP smoke test, no AWARE, seconds.
echo "=== step 2: band_critical smoke test ==="
python -m fsss.band_critical

# 3) clean, all five configs, full 400 iterations.
#    Watch: bit_acc ~1.0 everywhere, exp2b's coverage (should land 25-40%),
#    and the WSR column.
echo "=== step 3: exp_v16 clean, all configs ==="
python -m fsss.exp_v16_critical_decimation --clean-only --numtaps 4095

# 4) clean + METAPXYL proxy + key vox attacks.
echo "=== step 4: exp_v16 clean + attacks ==="
python -m fsss.exp_v16_critical_decimation --numtaps 4095

# 5) figures.
echo "=== step 5: figures ==="
python -m fsss.viz_critical --figs 1,2,3,4,5
python -m fsss.viz_critical --figs 6

echo "done $(date)"
