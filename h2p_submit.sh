#!/bin/bash -l
#SBATCH --cluster=smp
#SBATCH --partition=high-mem
#SBATCH --job-name=neuraldecoding
#SBATCH -o runs/outfiles/out.o%j
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mail-type=all
#SBATCH --mail-user=knoneman@pitt.edu
#SBATCH --time=0-01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --array=0-10

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load gcc/8.2.0
module load python/anaconda3.9-2021.11
conda activate neuraldecoding

echo "num cpus is $SLURM_CPUS_PER_TASK" # Size of multiprocessing pool

srun python decoding-2.py $1
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
