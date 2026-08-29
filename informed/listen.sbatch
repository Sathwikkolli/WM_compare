#!/bin/bash
#SBATCH --job-name=wmlisten
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16g
#SBATCH --time=00:30:00
#SBATCH --output=wmlisten_%j.log
#SBATCH --account=hafiz1

# --- Build the listening set --------------------------------------------------
#
# Generates audio at MEASURED quality levels so the usability floor can be
# judged by ear rather than taken on faith. Everything in the screen turns on
# that floor, and DNSMOS is a model, not a listening test.
#
# Batch rather than interactive: it scores several hundred clips and a dropped
# ssh connection would kill it partway.
#
# Afterwards, from your laptop:
#     scp -r <uniqname>@greatlakes-xfer.arc-ts.umich.edu:~/wm_compare/listen_floor .

export WM_COMPARE_BASE=$HOME/wm_compare
export EMILIA_CSV=${EMILIA_CSV:-/nfs/turbo/umd-hafiz/issf_server_data/emilia/manifests/emilia_curated.csv}
export TORCHDYNAMO_DISABLE=1
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate wmcompare

cd "$WM_COMPARE_BASE/informed"

if [ ! -f clips.json ]; then
    echo "ERROR: clips.json missing. Run 'python clips.py' first." >&2
    exit 1
fi

python listen_floor.py --out "$WM_COMPARE_BASE/listen_floor"

echo ""
echo "=============================================================="
echo " Copy these back and listen before trusting the floor:"
echo "   scp -r <uniqname>@greatlakes-xfer.arc-ts.umich.edu:$WM_COMPARE_BASE/listen_floor ."
echo ""
echo " Compare music_bed at a given level against gaussian_noise at the"
echo " SAME level. Same number, very different damage. If one is clearly"
echo " usable and the other is not, a single global floor is wrong and the"
echo " screen needs a per-attack one."
echo "=============================================================="
