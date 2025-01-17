#!/bin/bash

#SBATCH --partition=cpu-7d
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=256G

# Check if the script is being executed or submitted
if [ -z "$SLURM_JOB_ID" ]; then
    # Not running as an sbatch job, submit itself
    sbatch --output=$HOME/pml-bert/replibert/logs/baseline.log "$0" "$@"
    exit
fi

options="$@"

echo 'Running replibert baseline...'
apptainer run --bind /home/space/datasets:/home/space/datasets pml.sif python replibert/main.py baseline $options
echo 'Replibert baseline execution completed.'
