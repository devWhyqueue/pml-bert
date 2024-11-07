#!/bin/bash

srun --partition=cpu-2h --cpus-per-task=100 --mem-per-cpu=4G bash -c "
    echo 'Running replibert combine...'
    apptainer run --nv --bind /home/space/datasets:/home/space/datasets \
    pml.sif python replibert/main.py combine -d /home/space/datasets/wikipedia -d /home/space/datasets/project_gutenberg \
    --shuffle --keep text --destination ~/pml-bert/replibert/data/combined
    echo 'Replibert combine execution completed.'
"
