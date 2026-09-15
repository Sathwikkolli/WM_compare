#!/bin/bash
# informed/phase_b.sbatch -- NOT submitted directly. Use submit_phase_b.sh,
# which chains the stages with Slurm dependencies so nothing needs your
# ssh session to stay alive.
#
# Stage 0  prep      embed 200 null clips, cache org + w_clean   (single, slow)
# Stage 1  nullcal   threshold curves per attack, 3 arms         (array 0-21)
# Stage 2  sweep     bisect all arms per clip                    (array 0-49)
# Stage 3  curve     even-grid curves on clip 0, for gain_curve  (single)
# Stage 4  score     summaries + figures, registered AND corrected informed
#
# STAGE and PHASEB_RUN are set by submit_phase_b.sh.

#SBATCH --job-name=phaseb
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=02:00:00
#SBATCH --output=phaseb_%x_%A_%a.log
#SBATCH --account=hafiz1

export WM_COMPARE_BASE=$HOME/wm_compare
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}
export PHASEB_RUN=${PHASEB_RUN:-2026-08-28_informed-detection}
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare
cd "$WM_COMPARE_BASE/informed"

RES="$WM_COMPARE_BASE/results/$PHASEB_RUN"
STAGE=${STAGE:-sweep}
echo "=== Phase B stage: $STAGE   run: $PHASEB_RUN ==="

case "$STAGE" in
  prep)
      python null_calibrate.py --prep
      ;;

  # ATTACKS (optional, comma-separated) restricts every stage to those attacks.
  nullcal)
      if [ -n "${ATTACKS:-}" ]; then
          for A in ${ATTACKS//,/ }; do python null_calibrate.py --attack "$A" || exit 1; done
      else
          python null_calibrate.py
      fi
      ;;

  sweep)
      if [ ! -d "$RES/data/null" ]; then
          echo "ERROR: no null calibration in $RES. Run STAGE=nullcal first." >&2
          exit 1
      fi
      python bisect_sweep.py ${ATTACKS:+--attacks "$ATTACKS"}
      ;;

  curve)
      python bisect_sweep.py --clip 0 --attacks "${ATTACKS:-music_bed,lowpass,highpass}" --curve
      ;;

  score)
      for ARM in informed informed16; do
          echo ""
          echo "---- blind vs $ARM ----"
          python score_phase_b.py --informed-arm "$ARM" || echo "  (scoring $ARM failed)"
          python plots_phase_b.py --informed-arm "$ARM" || echo "  (plots $ARM failed)"
          python plots_phase_b.py --informed-arm "$ARM" --curve-attack lowpass \
              > /dev/null 2>&1 || true
      done
      echo ""
      # `|| true`: a glob matching nothing exits non-zero and, as the last
      # command, would mark a successful job FAILED.
      ls -la "$RES"/summary_phase_b*.md "$RES"/figures/*.png 2>/dev/null || true
      echo "score stage complete"
      exit 0
      ;;

  *)
      echo "ERROR: unknown STAGE '$STAGE' (prep|nullcal|sweep|curve|score)" >&2
      exit 1
      ;;
esac
