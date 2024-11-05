#!/bin/bash

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Submit the tokenize job array and capture its job ID
JOB_ID=$(sbatch --output=$HOME/pml-bert/replibert/logs/tokenize_%a.log --parsable "$SCRIPT_DIR/_tokenize_job.sh")

# Submit the unify job with dependency on the completion of the tokenize job
sbatch --dependency=afterok:$JOB_ID --output=$HOME/pml-bert/replibert/logs/tokenize.log <<'EOF'
#!/bin/bash
#SBATCH --partition=cpu-2h
#SBATCH --cpus-per-task=100
#SBATCH --mem-per-cpu=4G

echo 'Running replibert combine...'

# Construct the directory arguments
DIRS=""
for i in {0..99}; do
    DIRS="$DIRS -d ~/pml-bert/replibert/data/tokenized_tmp/$i"
done

apptainer run --nv --bind /temp:/temp pml.sif python replibert/main.py combine $DIRS \
    --destination ~/pml-bert/replibert/data/tokenized

rm -r ~/pml-bert/replibert/data/tokenized_tmp

echo 'Replibert combine execution completed.'
EOF
