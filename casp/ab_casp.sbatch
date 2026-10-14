#!/bin/bash
#SBATCH --job-name=ab_casp
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=ab_casp_%A_%a.log

# --- stock AWARE vs AWARE + CASP budget (same frozen detector), 5 clips,
# 3 attacks x 2 strengths + clean, with unwatermarked negatives.
#
#   cd $WM_COMPARE_BASE/casp
#   python ab_embed_detect.py --make-clips     # login node, once -> clips_ab.json
#   sbatch ab_casp.sbatch
#   python ab_aggregate.py                     # after the array finishes
#
# Needs results/2026-10-14_casp-threshold/<clip>/probe.npz for the 4 analysed
# clips (git pull). The 5th clip is probed inside its own task (~10 min extra).

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CASP_RUN=${CASP_RUN:-2026-10-14_casp-threshold}
export CASP_AB_RUN=${CASP_AB_RUN:-2026-10-14_casp-ab}

eval "$(conda shell.bash hook)"
conda activate wmcompare
command -v ffmpeg >/dev/null || module load ffmpeg     # mp3 attacks

cd "$WM_COMPARE_BASE/casp"
[ -f clips_ab.json ] || { echo "clips_ab.json missing: run 'python ab_embed_detect.py --make-clips'" >&2; exit 1; }

# fail early (not mid-run) if the env lacks something. `aware` is imported
# through masking/common.import_aware() in the real run, so only check the rest.
python - <<'PY' || exit 1
import importlib
for m in ["torch", "torchaudio", "librosa", "soundfile", "scipy", "pandas", "matplotlib", "pesq", "pystoi"]:
    importlib.import_module(m)
print("env ok")
PY

echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  task: $SLURM_ARRAY_TASK_ID"
python -u ab_embed_detect.py --index "$SLURM_ARRAY_TASK_ID" --threads "$SLURM_CPUS_PER_TASK"
