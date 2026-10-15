#!/bin/bash
#SBATCH --job-name=bm_ss
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=03:00:00
#SBATCH --array=0-39
#SBATCH --output=bm_ss_%A_%a.log

# --- SS vs AudioSeal / AWARE / Timbre. Tasks 0-19: ASVspoof clips20.json,
# tasks 20-39: Emilia test clips (seed 0). One clip per task.
#
#   cd $WM_COMPARE_BASE/casp
#   srun --account=hafiz1 --partition=standard --cpus-per-task=8 --mem=32g --time=00:30:00 \
#        python bm_run.py --selftest --dataset asv --index 0        # fails fast if anything is mis-wired
#   sbatch bm.sbatch
#   sbatch bm_null.sbatch
#   python bm_aggregate.py                                          # when both arrays finish

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export CASP_BENCH_RUN=${CASP_BENCH_RUN:-2026-10-14_casp-bench}
export CASP_RUN_DIR=${CASP_RUN_DIR:-$WM_COMPARE_BASE/results/2026-10-14_casp-threshold}

eval "$(conda shell.bash hook)"
conda activate wmcompare
command -v ffmpeg >/dev/null || module load ffmpeg

cd "$WM_COMPARE_BASE/casp"
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

T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 20 ]; then DS=asv; IDX=$T; else DS=emilia; IDX=$((T-20)); fi
echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  task $T -> $DS $IDX"
python -u bm_run.py --dataset "$DS" --index "$IDX" --targets 4.2
