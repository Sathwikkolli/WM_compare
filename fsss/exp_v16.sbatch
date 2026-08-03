#!/bin/bash
#SBATCH --job-name=v16_critical
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=/home/ksathwik/wm_compare/v16_%j.log
#
# exp1: critically-sampled subband embedding (fsss/band_critical.py).
#
# Cost shape: AWARE embeds by running a 400-iteration optimization PER FILE, so
# wall clock is dominated by (clips x configs) embeds, not by the attack grid.
# At 3 clips x 3 configs that is 9 embeds; the vox sweep afterwards is cheap by
# comparison because it only re-detects. All five configs run by default, so
# that is 3 clips x 5 = 15 embeds; --only trims the set.
#
# Steps 1-2 are the gate. They cost seconds and minutes respectively, and if
# either is wrong every number in step 4 is meaningless -- a bad chain produces
# a silently weak watermark, not an exception. Do not skip ahead.

eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

cd /home/ksathwik/wm_compare
echo "start $(date)"

# 1) DSP smoke test -- no AWARE, no GPU, a few seconds.
#    T1 split exactness > 200 dB, T2 decimation round trip > 40 dB,
#    T3 pipeline null > 40 dB, T5 must be far WORSE than T2, T6 must be ok.
#    If T2 is low the guard band is too narrow for the tap count.
echo "=== step 1: band_critical smoke test ==="
python -m fsss.band_critical

# 2) clean only -- runs verify() per config, then embeds with no attacks.
#    Expect clean bit_acc ~1.0 everywhere. Read verify()'s chain_null and
#    decimation_snr before trusting anything below.
echo "=== step 2: exp_v16 clean ==="
python -m fsss.exp_v16_critical_decimation --clean-only

# 3) figures that need no attack data -- confirms placement visually.
echo "=== step 3: figures F1-F5 ==="
python -m fsss.viz_critical --figs 1,2,3,4,5

# 4) full run: clean + the METAPXYL proxy chain + the key vox attacks.
echo "=== step 4: exp_v16 clean + attacks ==="
python -m fsss.exp_v16_critical_decimation

# 5) robustness curves from the CSV step 4 just wrote.
echo "=== step 5: figure F6 ==="
python -m fsss.viz_critical --figs 6

echo "done $(date)"
