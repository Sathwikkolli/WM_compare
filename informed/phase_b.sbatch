#!/bin/bash
# informed/phase_b.sbatch -- NOT submitted directly. Use submit_phase_b.sh,
# which chains the three stages with Slurm dependencies so nothing needs your
# ssh session to stay alive.
#
# Stage 0  prep      embed 200 null clips, cache org + w_clean   (single, slow)
# Stage 1  nullcal   threshold curves per attack                 (array 0-21)
# Stage 2  sweep     bisect both arms per clip                   (array 0-49)
# Stage 3  score     gains + paired test                         (single)
#
# STAGE is set by submit_phase_b.sh; running this file bare defaults to the
# sweep, which will fail loudly if Stage 1 has not run.

#SBATCH --job-name=phaseb
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24g
#SBATCH --time=02:00:00
#SBATCH --output=phaseb_%x_%A_%a.log
#SBATCH --account=hafiz1

# TIMING -- estimated, not measured.
#   prep     200 embeds at ~15 s = ~50 min. The slow stage, run once.
#   nullcal  12 strengths x 200 clips x 2 arms per attack. AWARE detect
#            dominates; ~40 min per attack is the working guess.
#   sweep    22 attacks x 2 arms x 12 evals = 528 evaluations per clip, ~5 min.
#   score    reads CSVs, minutes.
# TIME ONE OF EACH before submitting the arrays -- submit_phase_b.sh prints the
# commands.

export WM_COMPARE_BASE=$HOME/wm_compare
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare
cd "$WM_COMPARE_BASE/informed"

STAGE=${STAGE:-sweep}
echo "=== Phase B stage: $STAGE ==="

case "$STAGE" in
  prep)
      python null_calibrate.py --prep
      ;;

  nullcal)
      # One task per attack. SLURM_ARRAY_TASK_ID selects it.
      python null_calibrate.py
      ;;

  sweep)
      if [ ! -d "$WM_COMPARE_BASE/results/2026-08-28_informed-detection/data/null" ]; then
          echo "ERROR: no null calibration. Run STAGE=nullcal first." >&2
          exit 1
      fi
      python bisect_sweep.py
      ;;

  score)
      python score_phase_b.py || echo "  (scoring failed or had no data)"
      echo ""
      python plots_phase_b.py || echo "  (plots failed or had no data)"
      RES="$WM_COMPARE_BASE/results/2026-08-28_informed-detection"
      echo ""
      ls -la "$RES"/summary_phase_b.md "$RES"/data/phase_b_*.csv 2>/dev/null
      ls -la "$RES"/figures/*phase_b* "$RES"/figures/gain_*.png 2>/dev/null
      ;;

  *)
      echo "ERROR: unknown STAGE '$STAGE' (prep|nullcal|sweep|score)" >&2
      exit 1
      ;;
esac
