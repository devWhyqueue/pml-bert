#!/bin/bash

srun --partition=cpu-2h bash -c "
    echo 'Building the container from pml.def...'
    apptainer build pml.sif pml.def
    echo 'Container build completed: pml.sif'
"
