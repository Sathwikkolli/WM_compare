#!/bin/bash
#SBATCH --job-name=damage_ab
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=01:00:00
#SBATCH --output=damage_ab_%j.log
#SBATCH --account=hafiz1

# --- attack-damage control: 5 matched pairs x 15 conditions ------------------
#
# NO ARRAY, on purpose. 75 rows is ~25 min of CPU; an array of 5 would spend
# more time loading the AWARE checkpoint five times than it saves, and a single
# job writes one raw.csv instead of five that have to be concatenated.
#
# CPU partition: AWARE detection is fine on CPU for ~10 s clips and `standard`
# queues far sooner than `gpu`. Add --partition=gpu --gres=gpu:1 if you want it
# on a GPU; --account is required either way.

export WM_COMPARE_BASE=$HOME/wm_compare
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}
export DAMAGE_RUN=${DAMAGE_RUN:-2026-08-17_attack-damage-control}
export AWARE_MODEL=${AWARE_MODEL:-AWARE}

export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare

# module load ffmpeg     # uncomment if ffmpeg is not already on PATH.
                         # mp3_32k is the PESQ quality anchor -- without ffmpeg
                         # it records SKIP and the run has no reference point.

cd "$WM_COMPARE_BASE/damage_ab"

# Both built on the login node first: a bad Emilia path or a broken AWARE
# checkpoint should fail in ten seconds, interactively, not after a queue wait.
if [ ! -f clips.json ]; then
    echo "ERROR: clips.json missing. Run 'python make_pairs.py' first." >&2
    exit 1
fi
if [ ! -f work/payloads.json ]; then
    echo "ERROR: work/payloads.json missing. Run 'python embed_pairs.py' first." >&2
    exit 1
fi

python run_damage.py

# --- then, on a login node: --------------------------------------------------
#   conda activate wmcompare && cd $WM_COMPARE_BASE/damage_ab
#   python analyze.py        # -> results/<run>/summary.md + figures/
#
# and to listen (from your laptop, NOT on the cluster):
#   scp -r greatlakes:~/wm_compare/damage_ab/work/listen ./listen
# -----------------------------------------------------------------------------
