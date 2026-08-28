#!/bin/bash
#SBATCH --job-name=wmscreen
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=02:00:00
# TIMING -- estimated, NOT measured. Per clip: 1 AWARE embed, then
# 118 configs x 2 arms = 236 rows, each needing one attack, one AWARE detect,
# one DNSMOS and one PESQ. Estimate ~15-20 min/clip, dominated by encodec
# (a neural codec) and the ffmpeg round-trips.
#
# TIME ONE CLIP BEFORE SUBMITTING 50:
#     time python screen_sweep.py --clip 0
# If it exceeds ~40 min, either raise --time or split the grid by category:
#     sbatch --export=ALL,CATS=additive,acoustic screen.sbatch
#
# Memory is set for encodec plus the DNSMOS model held together; 32 g is
# deliberate headroom, since an OOM kills the whole task near the end.
#SBATCH --array=0-49
#SBATCH --output=wmscreen_%A_%a.log
#SBATCH --account=hafiz1

# --- Phase A: the 27-attack boundary screen -----------------------------------
#
# Which attacks break detection while the audio is still USABLE? That is the
# only kind of failure worth attacking with informed detection -- an attacker
# who destroys the audio has already lost.
#
# 2026-08-10_aware-detection-ab found two failures; 2026-08-17_attack-damage-
# control then showed both only work by wrecking the audio (PESQ 1.04 and 1.32
# against a clean 4.64). This screen looks for failures that survive that test.
#
# One array task per clip -> results/<run>/data/screen_clip<NNN>.csv.
# Re-running a task overwrites only that clip's file.

export WM_COMPARE_BASE=$HOME/wm_compare
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}

export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare        # + requirements_extra.txt

cd "$WM_COMPARE_BASE/informed"

if [ ! -f clips.json ]; then
    echo "ERROR: clips.json missing. Run 'python clips.py' first." >&2
    exit 1
fi

# THE QUALITY BACKEND GATE. The quality axis IS the screen -- without a working
# no-reference scorer this run produces detection numbers we already have and
# nothing new. quality.py checks the backend loads AND that its score falls
# monotonically with added noise; a backend that dips and recovers cannot locate
# a crossing. The SQUIM fallback failed exactly this check locally.
if ! python quality.py; then
    echo "ERROR: quality backend failed its self-test. Refusing to run." >&2
    echo "       pip install -r requirements_extra.txt" >&2
    exit 1
fi

# Report which attacks can actually run here. Missing ffmpeg / transformers /
# pydub silently turns attacks into SKIPs, and it is far better to see that in
# the first lines of the log than to discover a third of the grid is empty at
# scoring time.
python attacks_screen.py

if [ -n "$CATS" ]; then
    echo "restricting to categories: $CATS"
    python screen_sweep.py --categories "$CATS"
else
    python screen_sweep.py
fi

# --- after the array finishes, run once on a login node: ----------------------
#   conda activate wmcompare && cd $WM_COMPARE_BASE/informed
#   python score_screen.py
#
# That writes summary_screen.md, data/screen_metrics.csv, and
# selected_attacks.json -- the Phase B input. Phase B reads that file rather
# than hardcoding an attack list, because choosing the attacks by evidence is
# this screen's entire purpose.
#
# Then write the conclusion into
# results/2026-08-28_informed-detection/README.md BY HAND (results/README.md
# rule 2) and update the INDEX row.
# -----------------------------------------------------------------------------
