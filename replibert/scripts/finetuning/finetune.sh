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

# Set MASTER_ADDR to the hostname of the first node
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=12355

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

nvidia-smi -L

options="$@"

echo 'Running replibert with torchrun...'

# Use srun to launch one task per GPU and properly assign CUDA_VISIBLE_DEVICES
srun --ntasks-per-node=$SLURM_NTASKS --gpus-per-task=1 bash -c "
# Extract the correct GPU or MIG device for this task
GPU_LIST=(${CUDA_VISIBLE_DEVICES//,/ })
ASSIGNED_GPU=\${GPU_LIST[\$SLURM_LOCALID]}
export CUDA_VISIBLE_DEVICES=\$ASSIGNED_GPU
echo \"Task \$SLURM_PROCID running on GPU \$CUDA_VISIBLE_DEVICES\"

apptainer run --nv --bind /home/space/datasets:/home/space/datasets pml.sif \
    python replibert/main.py finetune $options
"

echo "Replibert execution completed."
