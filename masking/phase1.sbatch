#!/bin/bash
#SBATCH --job-name=mask_p1
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8g
#SBATCH --time=00:45:00
#SBATCH --array=0-29
#SBATCH --output=mask_p1_%A_%a.log

# --- Phase 1 (E2-E6): masking-threshold budget vs AWARE budget, noise stand-in.
# No AWARE embedding: STFT edits + PESQ/STOI + ffmpeg codecs only. Same 30 clips
# as E0/E1 (masking/clips.json, already on the cluster).
#
#   cd $WM_COMPARE_BASE/masking
#   python run_phase1.py --index 0     # optional smoke test, ~1-2 min
#   sbatch phase1.sbatch
#   python aggregate_phase1.py         # after the array finishes

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare
command -v ffmpeg >/dev/null || module load ffmpeg     # mp3 / opus need it

cd "$WM_COMPARE_BASE/masking"
[ -f clips.json ] || { echo "clips.json missing: run 'python make_clips.py' first" >&2; exit 1; }
echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  task: $SLURM_ARRAY_TASK_ID"
python run_phase1.py --index "$SLURM_ARRAY_TASK_ID"
