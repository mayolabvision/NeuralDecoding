#!/bin/bash -l
#SBATCH -p RM-shared
#SBATCH -J neuraldecoding
#SBATCH -o runs/outfiles/out.o%j
#SBATCH -N 1
#SBATCH --mail-type=all
#SBATCH --mail-user=knoneman@andrew.cmu.edu
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=4
#SBATCH --array=0-10

echo "My SLURM_ARRAY_JOB_ID is $SLURM_ARRAY_JOB_ID."
echo "My SLURM_ARRAY_TASK_ID is $SLURM_ARRAY_TASK_ID"
echo "Executing on the machine:" $(hostname)

module purge
module load anaconda3/2022.10
conda activate neuraldecoding

echo "num cpus is $SLURM_CPUS_PER_TASK" # Size of multiprocessing pool

python decoding-2.py $1

#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
