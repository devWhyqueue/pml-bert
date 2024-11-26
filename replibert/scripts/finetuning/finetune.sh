#!/bin/bash

#SBATCH --partition=gpu-5h
#SBATCH --gpus-per-node=10gb:1

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
