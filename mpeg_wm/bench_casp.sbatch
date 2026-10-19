#!/bin/bash
#SBATCH --job-name=casp_vox
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --array=0-119
#SBATCH --output=casp_vox_%A_%a.log

# --- CASP placement at MPEG's energy: CASP-1 (MPEG start), CASP-2 (flat start) and
# their shuffled controls, on the same 120 clips / Vox attacks as bench.sbatch.
# Results go to a NEW run folder; the earlier 4-system run is merged in at aggregation.
#
#   cd $WM_COMPARE_BASE && git pull && cd mpeg_wm
#   python test_casp.py                                             # ~15 s, no data
#   python bench_run.py --index 0 --selftest --no-wavs \
#       --systems casp1,casp1_shuf,casp2,casp2_shuf --out /tmp/casp_selftest   # timing check
#   sbatch --array=0-119%30 bench_casp.sbatch
#   MPEG_BENCH_RUN=2026-10-20_casp-vox-bench,2026-10-19_mpeg-vox-bench \
#   MPEG_BENCH_OUT=2026-10-20_casp-vox-bench python bench_aggregate.py

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$WM_COMPARE_BASE/mpeg_wm/work/numba_cache}
export MPEG_BENCH_RUN=${MPEG_BENCH_RUN:-2026-10-20_casp-vox-bench}

eval "$(conda shell.bash hook)"
conda activate wmcompare
command -v ffmpeg >/dev/null || module load ffmpeg

cd "$WM_COMPARE_BASE/mpeg_wm"
python - <<'PY' || exit 1
import importlib
for m in ["numpy", "scipy", "pandas", "librosa", "soundfile", "pesq", "pystoi", "numba"]:
    importlib.import_module(m)
for m in ["transformers", "julius", "pydub"]:
    try:
        importlib.import_module(m)
    except Exception as e:
        print("optional attack dependency missing:", m, e)
print("env ok")
PY

echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  task $SLURM_ARRAY_TASK_ID"
python -u bench_run.py --index "$SLURM_ARRAY_TASK_ID" --targets 4.2 \
    --systems casp1,casp1_shuf,casp2,casp2_shuf
