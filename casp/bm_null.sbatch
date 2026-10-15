#!/bin/bash
#SBATCH --job-name=bm_null
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=02:00:00
#SBATCH --array=0-9
#SBATCH --output=bm_null_%A_%a.log

# --- unwatermarked Emilia clips (seed 1, disjoint from the test clips) through
# every attack config; 10 chunks x 10 clips = 100 null clips.
export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TORCHDYNAMO_DISABLE=1 TORCH_COMPILE_DISABLE=1
export CASP_BENCH_RUN=${CASP_BENCH_RUN:-2026-10-15_casp-bench-v2}

eval "$(conda shell.bash hook)"
conda activate wmcompare
command -v ffmpeg >/dev/null || module load ffmpeg
cd "$WM_COMPARE_BASE/casp"
echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  chunk $SLURM_ARRAY_TASK_ID"
python -u bm_null.py --chunk "$SLURM_ARRAY_TASK_ID" --n-per-chunk 10
