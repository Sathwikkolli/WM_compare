#!/bin/bash
#SBATCH --job-name=casp_thr
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --array=0-3
#SBATCH --output=casp_thr_%A_%a.log

# --- CASP masking-threshold experiment, 4 clips (no AWARE embedding).
# One array task per clip, index = row in clips2.json (0 = most continuous speech,
# last = most pauses). Task 0 also runs the model validation (calibration tone,
# level growth, 16 vs 44.1 kHz).
#
#   cd $WM_COMPARE_BASE/casp
#   python select_clips.py             # login node, once: writes clips2.json
#   python run_casp.py --synth --quick # optional smoke test (~2-5 min)
#   sbatch casp_threshold.sbatch
#   # then: git add results/2026-10-14_casp-threshold && push; pull locally.

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CASP_RUN=${CASP_RUN:-2026-10-14_casp-threshold}

eval "$(conda shell.bash hook)"
conda activate wmcompare

cd "$WM_COMPARE_BASE/casp"
[ -f clips2.json ] || { echo "clips2.json missing: run 'python select_clips.py' first" >&2; exit 1; }
[ -d vendor/src/torch_amt ] || { echo "vendor/src/torch_amt missing: git pull the casp/ folder" >&2; exit 1; }

# fail early (not mid-run) if the env lacks something
python - <<'PY' || exit 1
import importlib, sys
need = ["torch", "torchaudio", "numpy", "scipy", "librosa", "soundfile", "matplotlib"]
opt = ["pesq", "pystoi"]
bad = []
for m in need:
    try:
        mod = importlib.import_module(m); print(m, getattr(mod, "__version__", "?"))
    except Exception as e:
        bad.append(f"{m}: {e}")
for m in opt:
    try: importlib.import_module(m); print(m, "ok")
    except Exception: print(m, "MISSING (PESQ/STOI columns will be empty)")
print("python", sys.version.split()[0])
if bad:
    print("MISSING:", *bad, sep="\n  "); sys.exit(1)
PY

echo "commit: $(git -C "$WM_COMPARE_BASE" rev-parse --short HEAD)  task: $SLURM_ARRAY_TASK_ID  run: $CASP_RUN"
RES="$WM_COMPARE_BASE/results/$CASP_RUN"
mkdir -p "$RES"

if [ "$SLURM_ARRAY_TASK_ID" = "0" ]; then
  python validate.py --fs-compare --threads "$SLURM_CPUS_PER_TASK" --out "$RES/validation"
fi
python -u run_casp.py --index "$SLURM_ARRAY_TASK_ID" --threads "$SLURM_CPUS_PER_TASK"
