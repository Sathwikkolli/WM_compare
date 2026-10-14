#!/bin/bash
#SBATCH --job-name=sbm_casp
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --array=0-19
#SBATCH --output=sbm_casp_%A_%a.log

# --- multibit payload sweep (1/4/8/20 bits) of the CASP-shaped spread-spectrum
# scheme vs real AWARE 6 dB, matched to AWARE's PESQ per clip, 20 clips x 10
# attacks + clean. Clips without a CASP probe.npz are probed inside their own
# task (about 10 extra minutes).
#
#   cd $WM_COMPARE_BASE/casp
#   python sb_multi.py --make-clips            # login node, once -> clips20.json
#   sbatch sb_multi.sbatch
#   python sb_multi_aggregate.py               # after the array finishes

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CASP_RUN=${CASP_RUN:-2026-10-14_casp-threshold}
export CASP_SBM_RUN=${CASP_SBM_RUN:-2026-10-14_casp-sbm}

eval "$(conda shell.bash hook)"
conda activate wmcompare
command -v ffmpeg >/dev/null || module load ffmpeg     # mp3 / opus attacks

cd "$WM_COMPARE_BASE/casp"
[ -f clips20.json ] || { echo "clips20.json missing: run 'python sb_multi.py --make-clips'" >&2; exit 1; }

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
python -u sb_multi.py --index "$SLURM_ARRAY_TASK_ID" --threads "$SLURM_CPUS_PER_TASK"
