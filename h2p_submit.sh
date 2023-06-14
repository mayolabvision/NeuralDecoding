#!/bin/bash -l
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=neuraldecoding
#SBATCH --output=runs/outfiles/out_%A_%a.out
#SBATCH --error=runs/outfiles/error_%A_%a.err
#SBATCH --nodes=1
#SBATCH --mail-type=fail
#SBATCH --mail-user=knoneman@pitt.edu
#SBATCH --time=0-24:00:00
#SBATCH --array=0-9

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load gcc/8.2.0
module load python/anaconda3.9-2021.11
conda activate neuraldecoding

python all_decoders.py $1
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
