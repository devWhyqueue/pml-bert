#!/bin/bash
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=64
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-9

# Calculate start and end indices based on the array task ID
start_index=$((SLURM_ARRAY_TASK_ID * 10))
end_index=$((start_index + 10))

echo "Running replibert tokenize for split indices $start_index to $end_index on node $HOSTNAME..."

apptainer run --bind /temp:/temp pml.sif python replibert/main.py tokenize \
--dataset_dir ~/pml-bert/replibert/data/combined \
--destination ~/pml-bert/replibert/data/tokenized_tmp --start_index $start_index --end_index $end_index

echo "Replibert tokenize execution completed for split indices $start_index to $end_index."
