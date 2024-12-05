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
echo "CUDA_VISIBLE_DEVICES before filtering: $CUDA_VISIBLE_DEVICES"

echo "Querying GPU information..."
gpu_map=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader)

# Parse GPU mapping and filter unique physical GPUs
declare -A physical_gpus
visible_devices=""
while IFS=',' read -r index uuid; do
    physical_gpu_id=${uuid%%-*}  # Extract physical GPU part from UUID
    if [[ -z "${physical_gpus[$physical_gpu_id]}" ]]; then
        physical_gpus[$physical_gpu_id]=1
        if [[ -z "$visible_devices" ]]; then
            visible_devices=$index
        else
            visible_devices="$visible_devices,$index"
        fi
    fi
done <<< "$gpu_map"

export CUDA_VISIBLE_DEVICES=$visible_devices
echo "CUDA_VISIBLE_DEVICES set to: $CUDA_VISIBLE_DEVICES"

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


