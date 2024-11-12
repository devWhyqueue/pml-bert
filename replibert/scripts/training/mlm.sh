#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Submit the mlm job array and capture its job ID
JOB_ID=$(sbatch --output=$HOME/pml-bert/replibert/logs/mlm_%a.log --parsable "$SCRIPT_DIR/_mlm_job.sh")

# Submit the combine job with dependency on the completion of the mlm job array
sbatch --dependency=afterok:$JOB_ID --output=$HOME/pml-bert/replibert/logs/mlm_combine.log <<'EOF'
#!/bin/bash
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=4G

echo 'Running replibert MLM combine...'

# Construct the directory arguments
DIRS=""
for i in {0..99}; do
    DIRS="$DIRS -d ~/pml-bert/replibert/data/bert_train_data_tmp/$i"
done

apptainer run --bind /home/space/datasets:/home/space/datasets pml.sif python replibert/main.py combine $DIRS \
    --destination /home/space/datasets/bert_train_data

# Clean up temporary directories
rm -r ~/pml-bert/replibert/data/bert_train_data_tmp

echo 'Replibert MLM combine execution completed.'
EOF
