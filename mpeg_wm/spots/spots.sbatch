#!/bin/bash
#SBATCH --job-name=casp_spots
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=03:00:00
#SBATCH --array=0-29
#SBATCH --output=casp_spots_%A_%a.log

# --- Where do CASP and the deployed MPEG watermark disagree? 30 clips = every 4th
# clip of mpeg_wm/clips120.json (indices 0, 4, ..., 116). One clip per task.
#
#   cd $WM_COMPARE_BASE && git pull && cd mpeg_wm/spots
#   python scan.py --index 0 --quick --workers 8 --out /tmp/spots_quick     # ~1-2 min plumbing check
#   sbatch spots.sbatch
#   python aggregate.py                                                     # when the array finishes

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1           # parallelism comes from the worker processes
export NUMBA_CACHE_DIR=${NUMBA_CACHE_DIR:-$WM_COMPARE_BASE/mpeg_wm/work/numba_cache}
export SPOTS_RUN=${SPOTS_RUN:-2026-10-22_casp-spots}

eval "$(conda shell.bash hook)"
conda activate wmcompare

cd "$WM_COMPARE_BASE/mpeg_wm/spots"
python -c "import numpy, scipy, numba, librosa, pandas; print('env ok')" || exit 1
IDX=$((SLURM_ARRAY_TASK_ID * 4))
echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  task $SLURM_ARRAY_TASK_ID -> clip index $IDX"
python -u scan.py --index "$IDX" --workers "$SLURM_CPUS_PER_TASK"
