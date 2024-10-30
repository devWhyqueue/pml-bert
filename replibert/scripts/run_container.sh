#!/bin/bash

echo "Requesting a GPU node on the 'gpu-test' partition with 1 GPU..."
srun --partition=gpu-test --gpus=1 bash -c "
    echo 'Running replibert inside the container...'
    apptainer run --nv --bind /home/space/datasets:/home/space/datasets pml.sif python replibert/main.py
    echo 'Replibert execution completed.'
"
