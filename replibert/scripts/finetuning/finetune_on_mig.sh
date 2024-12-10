#!/bin/bash

#SBATCH --partition=gpu-teaching-2d
#SBATCH --gpus-per-node=8
#SBATCH --nodes=1
#SBATCH --nodelist=head022
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=2

# Check if the script is being executed or submitted
if [ -z "$SLURM_JOB_ID" ]; then
    # Not running as an sbatch job, submit itself
    sbatch --output=$HOME/pml-bert/replibert/logs/finetune.log "$0" "$@"
    exit
fi

MASTER_ADDR=localhost
MASTER_PORT=29500

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "Original CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

options="$@"

echo "Parsing MIG devices to determine their parent GPUs..."

DEVICE_INFO=$(nvidia-smi -L)

# MIG_TO_GPU_MAP will map MIG-<uuid> to a GPU index number
declare -A MIG_TO_GPU_MAP

current_gpu_idx=""

# Parse nvidia-smi -L output
while IFS= read -r line; do
    # Match lines like: GPU 0: NVIDIA A100 ... (UUID: GPU-...)
    if [[ $line =~ ^GPU[[:space:]]([0-9]+): ]]; then
        current_gpu_idx="${BASH_REMATCH[1]}"
    fi

    # Match MIG lines like:
    #   MIG 1g.10gb     Device  0: (UUID: MIG-3c4c3356-0d98-56e4-8669-0efdaf333be5)
    if [[ $line =~ UUID:\ (MIG-[0-9a-fA-F-]+) ]]; then
        mig_uuid="${BASH_REMATCH[1]}"
        if [ -n "$current_gpu_idx" ]; then
            MIG_TO_GPU_MAP["$mig_uuid"]="$current_gpu_idx"
        else
            echo "Warning: Found MIG device $mig_uuid but no current GPU index!"
        fi
    fi
done <<< "$DEVICE_INFO"

IFS=',' read -ra ALL_DEVICES <<< "$CUDA_VISIBLE_DEVICES"

declare -A GPU_GROUPS
for dev in "${ALL_DEVICES[@]}"; do
    # dev is something like MIG-3c4c3356-0d98-56e4-8669-0efdaf333be5
    parent_gpu="${MIG_TO_GPU_MAP[$dev]}"
    if [ -n "$parent_gpu" ]; then
        # Append this device to that GPU's list
        GPU_GROUPS[$parent_gpu]+="$dev "
    else
        echo "Warning: Could not find parent GPU for device $dev. It will not be used."
    fi
done

num_tasks=$SLURM_NTASKS_PER_NODE
selected_devices=()

# Select at most one MIG device per distinct GPU for the tasks
for gpu_id in "${!GPU_GROUPS[@]}"; do
    devices=(${GPU_GROUPS[$gpu_id]})
    if [ ${#devices[@]} -gt 0 ]; then
        selected_devices+=("${devices[0]}")
    fi
    if [ ${#selected_devices[@]} -ge $num_tasks ]; then
        break
    fi
done

echo "Selected MIG devices for DDP: ${selected_devices[@]}"
DEVICES_CSV=$(IFS=','; echo "${selected_devices[*]}")
NUM_SELECTED_DEVICES=${#selected_devices[@]}

if [ ${#selected_devices[@]} -lt $num_tasks ]; then
    echo "Warning: Not enough distinct MIG GPUs found. Only ${#selected_devices[@]} out of $num_tasks tasks will get a device."
fi

echo 'Running replibert...'

srun --ntasks-per-node=$SLURM_NTASKS_PER_NODE bash -c "
    IFS=',' read -ra SEL_DEV <<< \"$DEVICES_CSV\"
    if (( SLURM_LOCALID < \${#SEL_DEV[@]} )); then
        ASSIGNED_GPU=\"\${SEL_DEV[\$SLURM_LOCALID]}\"
        export CUDA_VISIBLE_DEVICES=\$ASSIGNED_GPU
        echo \"Task \$SLURM_PROCID assigned MIG device \$CUDA_VISIBLE_DEVICES\"

        export RANK=\$SLURM_PROCID
        export WORLD_SIZE=$NUM_SELECTED_DEVICES
        export MASTER_ADDR=$MASTER_ADDR
        export MASTER_PORT=$MASTER_PORT

        apptainer run --nv --bind /home/space/datasets:/home/space/datasets pml.sif \
            python replibert/main.py finetune $options
    else
        echo \"Task \$SLURM_PROCID: No distinct MIG device available. Skipping replibert execution.\"
    fi
"


echo "Replibert execution completed."
