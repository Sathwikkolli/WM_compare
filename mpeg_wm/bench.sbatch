#!/bin/bash
#SBATCH --job-name=mpeg_vox
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --array=0-119
#SBATCH --output=mpeg_vox_%A_%a.log

# --- MPEG masking watermark vs AudioSeal / AWARE / Timbre, VoxWatermark attacks.
# One ASVspoof clip per task (mpeg_wm/clips120.json).
#
#   cd $WM_COMPARE_BASE && git pull && cd mpeg_wm
#   python make_clips.py                                            # login node, once
#   srun --account=hafiz1 --partition=standard --cpus-per-task=8 --mem=32g --time=00:45:00 \
#        python bench_run.py --index 0 --selftest                    # fails fast if mis-wired
#   sbatch bench.sbatch
#   python bench_aggregate.py                                       # when the array finishes

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export MPEG_BENCH_RUN=${MPEG_BENCH_RUN:-2026-10-19_mpeg-vox-bench}

eval "$(conda shell.bash hook)"
conda activate wmcompare
command -v ffmpeg >/dev/null || module load ffmpeg

cd "$WM_COMPARE_BASE/mpeg_wm"
python - <<'PY' || exit 1
import importlib
for m in ["numpy", "scipy", "pandas", "librosa", "soundfile", "pesq", "pystoi", "torch", "audioseal"]:
    importlib.import_module(m)
for m in ["transformers", "julius", "pydub"]:
    try:
        importlib.import_module(m)
    except Exception as e:
        print("optional attack dependency missing:", m, e)
print("env ok")
PY

echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  task $SLURM_ARRAY_TASK_ID"
python -u bench_run.py --index "$SLURM_ARRAY_TASK_ID" --targets 4.2
