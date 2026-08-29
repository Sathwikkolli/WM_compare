#!/bin/bash
# informed/submit_all.sh -- submit the whole of Phase A, unattended.
#
# Everything runs as a batch job and the scoring job WAITS for both arrays via a
# Slurm dependency, so nothing here needs your ssh session to stay alive. Submit
# it, disconnect, come back to finished summaries.
#
#     bash submit_all.sh
#
# Jobs submitted:
#   1. musicsweep   array 0-49   Phase A-1, the fine music SNR sweep
#   2. wmscreen     array 0-49   Phase A, all 27 attacks
#   3. wmlisten     single       the listening set for the usability floor
#   4. wmscore      single       scoring + plots, AFTER 1 and 2 both succeed
#
# 1, 2 and 3 are independent and run concurrently. 4 is gated on 1 and 2.
#
# If an array task fails, `afterok` holds the scoring job rather than scoring
# partial data. Requeue the failed task, or release the dependency deliberately
# with `scancel` + a manual `sbatch score.sbatch` once you have decided the
# partial data is worth scoring.
set -u

cd "$(dirname "$0")"

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}

# ---- preflight: fail here, in two seconds, not inside 100 array tasks -------
if [ ! -f clips.json ]; then
    echo "ERROR: clips.json missing. Run 'python clips.py' first." >&2
    exit 1
fi

echo "checking the quality backend..."
if ! python quality.py > /tmp/wm_quality_check.$$ 2>&1; then
    echo "ERROR: quality backend failed its self-test. Nothing submitted." >&2
    cat /tmp/wm_quality_check.$$ >&2
    rm -f /tmp/wm_quality_check.$$
    exit 1
fi
grep -E "^no-reference backend|^OK|^FAIL" /tmp/wm_quality_check.$$
rm -f /tmp/wm_quality_check.$$
echo ""

# ---- submit -----------------------------------------------------------------
MUSIC=$(sbatch --parsable music_sweep.sbatch)
echo "  music sweep   job $MUSIC   (array 0-49)"

SCREEN=$(sbatch --parsable screen.sbatch)
echo "  27-attack screen  job $SCREEN   (array 0-49)"

LISTEN=$(sbatch --parsable listen.sbatch)
echo "  listening set job $LISTEN"

SCORE=$(sbatch --parsable --dependency="afterok:${MUSIC}:${SCREEN}" score.sbatch)
echo "  scoring       job $SCORE   (waits for $MUSIC and $SCREEN)"

echo ""
echo "=============================================================="
echo " All submitted. You can disconnect."
echo ""
echo "   squeue -u \$USER                 # what is still running"
echo "   sacct -j $SCORE --format=JobID,State,Elapsed"
echo ""
echo " When $SCORE finishes, read:"
echo "   \$WM_COMPARE_BASE/results/2026-08-28_informed-detection/summary_phase_a1.md"
echo "   \$WM_COMPARE_BASE/results/2026-08-28_informed-detection/summary_screen.md"
echo "   informed/selected_attacks.json     <- the Phase B input"
echo "=============================================================="
