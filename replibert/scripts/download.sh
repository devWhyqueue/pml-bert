#!/bin/bash

srun --partition=cpu-2h bash -c "
    echo 'Running replibert download...'
    apptainer run --nv --bind /home/space/datasets:/home/space/datasets pml.sif python replibert/main.py download
    echo 'Replibert download execution completed.'
"