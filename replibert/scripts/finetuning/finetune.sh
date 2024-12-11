#!/bin/bash

#SBATCH --partition=gpu-teaching-2h
#SBATCH --gpus-per-node=4
#SBATCH --nodes=1
#SBATCH --nodelist=head025
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4

# Check if the script is being executed or submitted
if [ -z "$SLURM_JOB_ID" ]; then
    # Not running as an sbatch job, submit itself
    sbatch --output=$HOME/pml-bert/replibert/logs/finetune.log "$0" "$@"
    exit
fi

echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

options="$@"

echo 'Running replibert...'

srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE bash -c "
    export CUDA_VISIBLE_DEVICES=\$SLURM_PROCID
    echo \"Task \$SLURM_PROCID assigned GPU \$CUDA_VISIBLE_DEVICES\"

    export RANK=\$SLURM_PROCID
    export WORLD_SIZE=\$SLURM_NTASKS_PER_NODE
    export MASTER_ADDR=localhost
    export MASTER_PORT=29500

    apptainer run --nv --bind /home/space/datasets:/home/space/datasets pml.sif \
        python replibert/main.py finetune $options
"

echo "Replibert execution completed."
