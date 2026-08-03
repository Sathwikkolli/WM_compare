#!/bin/bash
#SBATCH --job-name=v16_wsr
#SBATCH --account=hafiz1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ksathwik/wm_compare/v16wsr_%j.log
#
# ONE QUESTION: is exp1a's PESQ collapse (2.32 vs stock's 4.23) a property of
# critical sampling, or is it the host-scale mismatch?
#
# WHY THIS IS ITS OWN JOB
# The tol-6 run showed exp1a beating stock on five attack families. That result
# is only meaningful if the two were embedded at comparable loudness, and they
# were not: verify() reported budget_gap_db = 18.0. tolerance_db is applied to
# the HOST, and for the strip arms AWARE peak-normalises that host -- an
# isolated band ~18 dB quieter than the mix -- while the residual `lo` stays at
# original scale. So +/-6 dB on the host is not +/-6 dB on the output.
#
# chain_embed.py predicted exactly this and deferred it:
#   "the budget is not calibrated to the original masker. Fixing that is its own
#    step; budget_gap_db below measures how wrong it is."
#
# Clean only, no attacks, one tolerance. Minutes, not hours. Settle this before
# spending another job on the 27-embed tolerance sweep -- sweeping tol to make
# the PESQ numbers line up would be hand-compensating for a scale error.
#
# exp2a/exp2b now run by default. exp2a routes through the SAME
# normalise-host-then-add-raw-lo path in chain_embed.py, so if the mismatch is
# real it should show there too -- which would confirm it is inherited from the
# band_steer chain and not something critical decimation introduced. If exp2a
# instead SKIPs on the BandPlan constructor (the shifted band leaving (0, sr/2)
# for a strip this narrow), that is its own useful answer.
#
# WHAT TO READ -- the `wsr` column, nothing else:
#
#   stock  wsr ~= X dB
#   exp1a  wsr ~= X + 18 dB   -> scale mismatch confirmed. The PESQ gap and
#                               every robustness win in the last run are
#                               loudness. Fix: rescale the watermarked host back
#                               to the raw host's level before synthesis, in
#                               embed() and _through_chain(). Then re-run.
#   exp1a  wsr ~= X           -> I am wrong. The PESQ gap is genuinely about
#                               concentrating distortion into 600 Hz of the F2
#                               region, and the tolerance sweep is the right
#                               next step after all.

eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/ksathwik/wm_compare
echo "start $(date)"
python -c "import torch; print('cuda', torch.cuda.is_available())"

python -m fsss.exp_v16_critical_decimation \
    --clean-only \
    --numtaps 4095

echo "done $(date)"
