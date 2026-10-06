#!/bin/bash
#SBATCH --job-name=wm_spec
#SBATCH --account=hafiz1
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/home/ksathwik/wm_compare/deepfake_probe/wm_spec_%j.log
# Needs run_probe.sbatch to have finished (it writes the clean/aware_only wavs and
# clips.csv). AudioSeal/Timbre solo embeds are fast on CPU. Chain it with:
#   jid=$(sbatch --parsable deepfake_probe/run_probe.sbatch)
#   sbatch --dependency=afterok:$jid deepfake_probe/spectrograms.sbatch

eval "$(conda shell.bash hook)"; conda activate wmcompare
export TORCHDYNAMO_DISABLE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

cd /home/ksathwik/wm_compare/deepfake_probe
echo "start $(date)"
python spectrograms.py "$@"
echo "done $(date)"
