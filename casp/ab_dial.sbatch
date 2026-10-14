#!/bin/bash
#SBATCH --job-name=ab_dial
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24g
#SBATCH --time=03:00:00
#SBATCH --array=0-4
#SBATCH --output=ab_dial_%A_%a.log

# --- dial sweep: 5 AWARE tolerances + 5 CASP targets per clip, harder attacks,
# audibility of the real watermark measured with CASP. 10 embeds per clip.
#
#   cd $WM_COMPARE_BASE/casp
#   python ab_embed_detect.py --make-clips     # once, if clips_ab.json is missing
#   sbatch ab_dial.sbatch
#   python ab_dial_aggregate.py                # after the array finishes
#
# Needs results/2026-10-14_casp-threshold/<clip>/probe.npz (D grids; git pull
# or already on the cluster from the earlier runs).

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CASP_RUN=${CASP_RUN:-2026-10-14_casp-threshold}
export CASP_DIAL_RUN=${CASP_DIAL_RUN:-2026-10-14_casp-dial}

eval "$(conda shell.bash hook)"
conda activate wmcompare
command -v ffmpeg >/dev/null || module load ffmpeg     # mp3 / opus attacks

cd "$WM_COMPARE_BASE/casp"
[ -f clips_ab.json ] || { echo "clips_ab.json missing: run 'python ab_embed_detect.py --make-clips'" >&2; exit 1; }

python - <<'PY' || exit 1
import importlib, shutil, subprocess
for m in ["torch", "torchaudio", "librosa", "soundfile", "scipy", "pandas", "matplotlib", "pesq", "pystoi"]:
    importlib.import_module(m)
enc = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
for c in ("libmp3lame", "libopus"):
    assert c in enc, f"ffmpeg lacks {c}"
print("env ok")
PY

echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  task: $SLURM_ARRAY_TASK_ID"
python -u ab_dial.py --index "$SLURM_ARRAY_TASK_ID" --threads "$SLURM_CPUS_PER_TASK"
