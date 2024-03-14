#!/bin/bash -l
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=neuraldecoding
#SBATCH --error=outfiles/error_%A_%a.err
#SBATCH --output=outfiles/out_%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mail-type=fail
#SBATCH --mail-user=knoneman@pitt.edu
#SBATCH --time=0-09:29:59
#SBATCH --array=0-999

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"

module purge
module load gcc/8.2.0
module load python/anaconda3.9-2021.11
conda activate decoding

python etra_runs.py $1 1
#python neuron_sweeps.py $1 1
#python cross_decoders.py $1 1 

echo "DONE"
