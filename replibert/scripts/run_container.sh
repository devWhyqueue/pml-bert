#!/bin/bash

echo "Requesting a GPU node on the 'gpu-test' partition with 1 GPU..."
srun --partition=gpu-test --gpus=1 bash -c "
    echo 'Checking GPU availability in the container...'
    apptainer run --nv pml.sif python -c \"import torch; print('CUDA available:', torch.cuda.is_available())\"

    echo 'Running your Python script inside the container...'
    apptainer run --nv pml.sif python replibert/main.py

    echo 'Python script execution completed.'
"
