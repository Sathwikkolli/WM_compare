#!/bin/bash
#SBATCH --job-name=wmscore
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --output=wmscore_%j.log
#SBATCH --account=hafiz1

# --- Phase A scoring: turn the raw CSVs into verdicts -------------------------
#
# Runs as a BATCH JOB rather than on a login node so a dropped ssh connection
# cannot kill it. Submit it with a dependency on the two arrays and the whole
# pipeline completes unattended:
#
#     sbatch --dependency=afterok:<music_jobid>:<screen_jobid> score.sbatch
#
# `submit_all.sh` wires that up for you.
#
# Scoring is cheap -- it reads CSVs and interpolates. 30 min is generous.

export WM_COMPARE_BASE=$HOME/wm_compare
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare

cd "$WM_COMPARE_BASE/informed"

RES="$WM_COMPARE_BASE/results/2026-08-28_informed-detection"

echo "=============================================================="
echo " raw data present:"
echo "   music_clip*.csv   : $(ls -1 $RES/data/music_clip*.csv 2>/dev/null | wc -l) / 50"
echo "   screen_clip*.csv  : $(ls -1 $RES/data/screen_clip*.csv 2>/dev/null | wc -l) / 50"
echo "=============================================================="
echo ""

# Each scorer is run independently: if one array failed and the other did not,
# we still want the half that worked. `|| true` keeps a missing input from
# aborting the whole job.

echo "### Phase A-1: the music sweep ###############################"
python score_music_sweep.py || echo "  (music sweep scoring failed or had no data)"
echo ""
python plots_music_sweep.py || echo "  (music sweep plots failed or had no data)"
echo ""

echo "### Phase A: the 27-attack screen ############################"
python score_screen.py || echo "  (screen scoring failed or had no data)"
echo ""

echo "=============================================================="
echo " outputs"
echo "=============================================================="
ls -la "$RES"/summary_*.md 2>/dev/null
ls -la "$RES"/data/*metrics*.csv 2>/dev/null
ls -la "$RES"/figures/*.png 2>/dev/null
ls -la "$WM_COMPARE_BASE/informed/selected_attacks.json" 2>/dev/null

echo ""
echo "Read summary_phase_a1.md and summary_screen.md, then write the"
echo "conclusion into $RES/README.md BY HAND (results/README.md rule 2)."
