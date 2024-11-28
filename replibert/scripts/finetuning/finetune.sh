#!/bin/bash

#SBATCH --partition=gpu-teaching-2h
#SBATCH --gpus-per-node=2
#SBATCH --gres-flags=enforce-binding
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16

# Check if the script is being executed or submitted
if [ -z "$SLURM_JOB_ID" ]; then
    # Not running as an sbatch job, submit itself
    sbatch --output=$HOME/pml-bert/replibert/logs/finetune.log "$0" "$@"
    exit
fi

# Number of GPUs per node (specified by --gres)
GPUS_PER_NODE=2
NODE_COUNT=$SLURM_JOB_NUM_NODES
# Set MASTER_ADDR to the hostname of the first node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=12355

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "NODE_COUNT: $NODE_COUNT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi -L

options="$@"

echo 'Running replibert with torchrun...'

apptainer run --nv --bind /home/space/datasets:/home/space/datasets pml.sif \
    torchrun \
    --nnodes=$NODE_COUNT \
    --nproc_per_node=$GPUS_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    replibert/main.py finetune $options

echo 'Replibert execution completed.'


