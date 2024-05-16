#!/bin/bash -l
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=neuraldecoding
#SBATCH --error=/ix1/pmayo/decoding/outfiles/error_%A_%a.err
#SBATCH --output=/ix1/pmayo/decoding/outfiles/out_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mail-type=fail
#SBATCH --mail-user=knoneman@pitt.edu
#SBATCH --time=0-23:59:59
#SBATCH --array=0-99

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "My SLURM_ARRAY_TASK_COUNT is $SLURM_ARRAY_TASK_COUNT"

module purge
module load gcc/8.2.0
module load python/anaconda3.9-2021.11
conda activate decoding

#python full_runs.py $1 1
python etra_runs.py $1 1 

echo "DONE"
