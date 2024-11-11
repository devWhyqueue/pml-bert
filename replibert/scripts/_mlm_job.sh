#!/bin/bash
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-9

# Calculate start and end indices based on the array task ID
start_index=$((SLURM_ARRAY_TASK_ID * 10))
end_index=$((start_index + 10))

echo "Running replibert MLM for split indices $start_index to $end_index on node $HOSTNAME..."

apptainer run --nv --bind /home/space/datasets:/home/space/datasets \
pml.sif python replibert/main.py mlm \
    --dataset_dir ~/pml-bert/replibert/data/tokenized \
    --destination ~/pml-bert/replibert/data/bert_train_data_tmp \
    --start_index $start_index --end_index $end_index

echo "Replibert MLM execution completed for split indices $start_index to $end_index."
