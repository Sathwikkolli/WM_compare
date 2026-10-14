#!/bin/bash
#SBATCH --job-name=sb_casp
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8g
#SBATCH --time=01:30:00
#SBATCH --array=0-4
#SBATCH --output=sb_casp_%A_%a.log

# --- single-bit spread-spectrum watermark, 4 budget arms x 15 strengths x 10
# attacks, key-only correlation detector. No AWARE involved. 5 clips.
#
#   cd $WM_COMPARE_BASE/casp
#   python ab_embed_detect.py --make-clips     # once, if clips_ab.json is missing
#   sbatch sb.sbatch
#   python sb_aggregate.py                     # after the array finishes
#
# Needs results/2026-10-14_casp-threshold/<clip>/probe.npz (D grids).

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CASP_RUN=${CASP_RUN:-2026-10-14_casp-threshold}
export CASP_SB_RUN=${CASP_SB_RUN:-2026-10-14_casp-sb}

eval "$(conda shell.bash hook)"
conda activate wmcompare
command -v ffmpeg >/dev/null || module load ffmpeg     # mp3 / opus attacks

cd "$WM_COMPARE_BASE/casp"
[ -f clips_ab.json ] || { echo "clips_ab.json missing: run 'python ab_embed_detect.py --make-clips'" >&2; exit 1; }

python - <<'PY' || exit 1
import importlib, subprocess
for m in ["numpy", "scipy", "pandas", "librosa", "soundfile", "matplotlib", "pesq", "pystoi", "torch"]:
    importlib.import_module(m)
enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
for c in ("libmp3lame", "libopus"):
    assert c in enc, f"ffmpeg lacks {c}"
print("env ok")
PY

echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  task: $SLURM_ARRAY_TASK_ID"
python -u sb_run.py --index "$SLURM_ARRAY_TASK_ID" --threads "$SLURM_CPUS_PER_TASK"
