#!/bin/bash

echo "Requesting a CPU node on the 'cpu-2h' partition..."
srun --partition=cpu-2h bash -c "
    echo 'Building the container from pml.def...'
    apptainer build pml.sif pml.def

    echo 'Container build completed: pml.sif'
"
