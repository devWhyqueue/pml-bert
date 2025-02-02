#!/bin/bash

partition="gpu-teaching-9m"
dataset_dir_bind="/home/space/datasets:/home/space/datasets"

srun --partition=$partition --nodelist=head025 --gpus-per-node=1 \
      apptainer run --nv --bind $dataset_dir_bind pml.sif python replibert/main.py "$@"
