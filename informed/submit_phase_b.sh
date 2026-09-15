#!/bin/bash
# informed/submit_phase_b.sh -- submit all of Phase B, unattended.
#
# Stages chained with Slurm `afterok`, so a failure HOLDS the chain rather than
# scoring partial data. Submit, disconnect, come back to a finished summary.
#
#     bash submit_phase_b.sh --from nullcal   # the v2 re-run (null cache reused)
#     bash submit_phase_b.sh                  # everything, including prep
#
# Results go to results/$PHASEB_RUN. The default is the v2 re-run directory, so
# the registered first run in 2026-08-28_informed-detection is never overwritten.
set -u
cd "$(dirname "$0")"

export WM_COMPARE_BASE=${WM_COMPARE_BASE:-$HOME/wm_compare}
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}
export PHASEB_RUN=${PHASEB_RUN:-2026-09-15_informed-detection-v2}

FROM="prep"
if [ "${1:-}" = "--from" ]; then FROM="${2:-prep}"; fi

# ---- preflight: fail in seconds, not inside 70 array tasks ------------------
if [ ! -f clips.json ]; then
    echo "ERROR: clips.json missing. Run 'python clips.py' first." >&2
    exit 1
fi
if [ "$FROM" != "prep" ] && [ ! -f "$WM_COMPARE_BASE/real_audio/null_cache.npz" ]; then
    echo "ERROR: real_audio/null_cache.npz missing -- start with --from prep." >&2
    exit 1
fi

echo "validating the informed detector..."
if ! python informed_detector.py > /tmp/wm_id_check.$$ 2>&1; then
    echo "ERROR: informed_detector self-test FAILED. Nothing submitted." >&2
    tail -25 /tmp/wm_id_check.$$ >&2
    rm -f /tmp/wm_id_check.$$
    exit 1
fi
grep -E "SELF-TEST|FIR beats|score_16k" /tmp/wm_id_check.$$
rm -f /tmp/wm_id_check.$$

N_ATTACKS=$(python -c "import sys; sys.path.insert(0,'.'); import strength_axis as SA; print(len(SA.AXIS))")
N_CLIPS=$(python -c "import json; print(len(json.load(open('clips.json'))['clips']))")
echo "  $N_ATTACKS attacks with a strength axis, $N_CLIPS clips"
echo "  results -> results/$PHASEB_RUN"
echo ""

DEP=""
LAST=""

submit () {   # submit <stage> <extra sbatch args...>
    local stage="$1"; shift
    local jid
    if [ -n "$DEP" ]; then
        jid=$(sbatch --parsable --dependency="afterok:${DEP}" \
                     --export=ALL,STAGE="$stage",PHASEB_RUN="$PHASEB_RUN" \
                     --job-name="pb_$stage" "$@" phase_b.sbatch)
    else
        jid=$(sbatch --parsable --export=ALL,STAGE="$stage",PHASEB_RUN="$PHASEB_RUN" \
                     --job-name="pb_$stage" "$@" phase_b.sbatch)
    fi
    echo "  $stage  job $jid ${DEP:+(after $DEP)}"
    DEP="$jid"
    LAST="$jid"
}

case "$FROM" in
  prep)    STAGES="prep nullcal sweep curve score" ;;
  nullcal) STAGES="nullcal sweep curve score" ;;
  sweep)   STAGES="sweep curve score" ;;
  curve)   STAGES="curve score" ;;
  score)   STAGES="score" ;;
  *) echo "ERROR: --from must be prep|nullcal|sweep|curve|score" >&2; exit 1 ;;
esac

# Times raised for the third arm (informed16) and the monotonicity probes.
echo "submitting: $STAGES"
for s in $STAGES; do
    case "$s" in
      prep)    submit prep    --time=02:00:00 ;;
      nullcal) submit nullcal --time=04:00:00 --array=0-$((N_ATTACKS - 1)) ;;
      sweep)   submit sweep   --time=03:00:00 --array=0-$((N_CLIPS - 1)) ;;
      curve)   submit curve   --time=01:00:00 ;;
      score)   submit score   --time=00:45:00 ;;
    esac
done

RES="\$WM_COMPARE_BASE/results/$PHASEB_RUN"
echo ""
echo "=============================================================="
echo " Submitted. You can disconnect."
echo ""
echo "   squeue -u \$USER"
echo "   sacct -j $LAST --format=JobID,JobName%12,State,Elapsed"
echo ""
echo " When $LAST finishes:"
echo "   $RES/summary_phase_b.md             <- registered informed"
echo "   $RES/summary_phase_b_informed16.md  <- corrected informed"
echo "   $RES/figures/                       <- both figure sets"
echo "=============================================================="
