#!/bin/bash
#SBATCH --job-name=wmlisten2
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=00:40:00
#SBATCH --output=wmlisten2_%j.log
#SBATCH --account=hafiz1

# --- Render the audio behind every verdict in the screen ----------------------
#
# Every SECURE / VULNERABLE / NO_FLOOR call turns on whether the audio was still
# usable at that setting, and that judgement was made by a model. This renders
# the audio at exactly those settings so it can be checked by ear.
#
# Batch, not interactive: it loads AWARE, embeds, and runs up to ~85 attacks.
# A dropped ssh would kill it partway.
#
# Set CLIP to hear a different voice (the verdicts are means over 50 clips, so
# one clip is illustrative, not decisive):
#     sbatch --export=ALL,CLIP=7 listen_screen.sbatch
#
# Set ALL=1 to render every swept setting rather than s* and its neighbours:
#     sbatch --export=ALL,ALL=1 listen_screen.sbatch

export WM_COMPARE_BASE=$HOME/wm_compare
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare

cd "$WM_COMPARE_BASE/informed"

if [ ! -f "$WM_COMPARE_BASE/results/2026-08-28_informed-detection/data/screen_metrics.csv" ]; then
    echo "ERROR: screen_metrics.csv missing. Run score_screen.py first." >&2
    exit 1
fi

CLIP=${CLIP:-0}
EXTRA=""
if [ -n "${ALL:-}" ]; then EXTRA="--all"; fi

python listen_screen.py --clip "$CLIP" --out "$WM_COMPARE_BASE/listen_screen" $EXTRA

echo ""
echo "=============================================================="
echo " Copy back and listen:"
echo "   scp -r <uniqname>@greatlakes-xfer.arc-ts.umich.edu:$WM_COMPARE_BASE/listen_screen ."
echo ""
echo " Start with the files marked 'mildest'. Those are attacks where NOT ONE"
echo " setting was judged usable -- including music at 20 dB SNR, where the"
echo " music is barely audible. If those sound fine to you, the quality floor"
echo " is wrong for background-addition attacks and those verdicts change."
echo "=============================================================="
