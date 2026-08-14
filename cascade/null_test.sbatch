#!/bin/bash
#SBATCH --job-name=null_test
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/ksathwik/wm_compare/null_test_%j.log

# --- Do the detectors fire on audio with no watermark? -----------------------
#
# Detection only -- no embedding -- so this is one forward pass per model per
# file and runs on CPU. AWARE's slow path is its 400-iteration embed optimiser,
# which is not exercised here.
#
# Only AudioSeal and AWARE load: Timbre has no detector, so no Timbre repo or
# checkpoint is needed.
#
# 2 h covers ~300 Emilia clips with headroom. Short jobs schedule sooner.

eval "$(conda shell.bash hook)"
conda activate wmcompare

export WM_COMPARE_BASE=$HOME/wm_compare
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Override if the manifest has moved.
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}

cd "$WM_COMPARE_BASE/cascade"

echo "=== started $(date) ==="
echo "host: $(hostname)"

# Stage 1: the curated set, small and fast -- confirms the adapters load and the
# public-domain / synthetic files behave before spending time on 300 clips.
python null_test.py \
    --dir "$WM_COMPARE_BASE/audio/clean_set" \
    --repeat 2 \
    --out "$WM_COMPARE_BASE/cascade/null_curated.csv"

echo ""
echo "=== curated set done, starting Emilia sweep $(date) ==="

# Stage 2: enough clips to support a rate, not just an anecdote. Zero false
# positives in 300 supports "<1%" at 95% confidence (rule of three); 12 files
# could only support "<25%".
python null_test.py \
    --emilia 300 \
    --out "$WM_COMPARE_BASE/cascade/null_emilia.csv"

echo "=== done $(date) ==="
