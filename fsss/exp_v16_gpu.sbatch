#!/bin/bash
#SBATCH --job-name=v16_critical_gpu
#SBATCH --account=hafiz1
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=/home/ksathwik/wm_compare/v16gpu_%j.log
#
# GPU version of fsss/exp_v16.sbatch.
#
# WHAT THE GPU ACTUALLY BUYS
# AWARE embeds by running a 400-iteration optimization per file, and every
# iteration is a detector forward+backward plus (for exp1) two FFT-domain
# filter passes at full rate. That is exactly the shape GPUs accelerate, and
# it is the entire wall clock -- detection and the attack grid are cheap.
#
# The CPU-side cost that does NOT go away is vox_attacks: mp3/aac/opus shell
# out to ffmpeg, and encodec runs its own model. Hence 8 CPUs alongside the GPU
# rather than 1.
#
# PARTITION NOTE
# Great Lakes has both `gpu` (V100/A100-class, strong fp64) and `spgpu` (A40,
# fp64 is heavily rate-limited). The optimization loop runs in float32 so
# either is fine; only the once-per-file analysis in embed()/verify() is
# float64, and that is milliseconds. Swap to `--partition=spgpu` if the gpu
# queue is long.
#
# CHECK THE FIRST LINE OF STEP 2's OUTPUT. It prints the device. A GPU job that
# silently fell back to CPU produces identical numbers and only differs in wall
# clock, so this is the one thing to eyeball before letting the job run out.

eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /home/ksathwik/wm_compare
echo "start $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "no nvidia-smi"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available())"

# 1) DSP smoke test -- CPU only, seconds. Unchanged by the GPU move; run it so
#    the log is self-contained.
echo "=== step 1: band_critical smoke test ==="
python -m fsss.band_critical

# 2) clean, full 400 iterations, 3 clips, sharpened filter.
#    numtaps 4095: at 1023 the decimation round trip read 27 dB on speech and
#    the |H|^2 vs |H|^4 transition mismatch was the cause; 4x the taps bought
#    +6.5 dB. Clean bit_acc was 1.000 either way, but keep the margin.
#    THE NUMBER TO WATCH IS PESQ, not bit_acc -- at 60 iterations exp1a was
#    already down at 2.75 against stock's 4.17, and push_extremes keeps growing
#    the perturbation after the bits are correct, so 400 should be worse.
echo "=== step 2: exp_v16 clean, full iterations ==="
python -m fsss.exp_v16_critical_decimation --clean-only --numtaps 4095

# 3) figures that need no attack data.
echo "=== step 3: figures F1-F5 ==="
python -m fsss.viz_critical --figs 1,2,3,4,5

# 4) clean + METAPXYL proxy + key vox attacks, ACROSS TOLERANCES.
#    The tol-6 run gave mean PESQ stock 4.23 / exp1b 3.23 / exp1a 2.32, and
#    exp1a beat stock on six attack families -- while being ~1.9 PESQ louder.
#    That comparison is worthless as stated, because robustness tracks loudness.
#    Sweeping QUIETER (higher tol) walks exp1a up the PESQ ladder until it meets
#    stock, and only there does "does criticality help" have an answer.
#    Cost is linear in the number of tolerances: 3 tols x 3 clips x 3 configs
#    = 27 embeds. Trim to --tols 9,12 if the queue is tight.
echo "=== step 4: exp_v16 clean + attacks, tolerance sweep ==="
python -m fsss.exp_v16_critical_decimation --numtaps 4095 --tols 6,9,12

# 5) robustness curves from step 4's CSV.
echo "=== step 5: figure F6 ==="
python -m fsss.viz_critical --figs 6

echo "done $(date)"
