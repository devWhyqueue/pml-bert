#!/bin/bash

#SBATCH --partition=cpu-2h

# Check if the script is being executed or submitted
if [ -z "$SLURM_JOB_ID" ]; then
    # Not running as an sbatch job, submit itself
    sbatch --output=$HOME/pml-bert/replibert/logs/run.log "$0" "$@"
    exit
fi

options="$@"

echo 'Running replibert...'
apptainer run --bind /home/space/datasets:/home/space/datasets pml.sif python replibert/main.py $options
echo 'Replibert execution completed.'
