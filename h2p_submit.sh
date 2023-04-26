#!/bin/bash -l
#SBATCH --cluster=smp
#SBATCH --partition=smp
#SBATCH --job-name=neuraldecoding
#SBATCH -o runs/outfiles/out.o%j
#SBATCH --nodes=1
#SBATCH --mail-type=all
#SBATCH --mail-user=knoneman@pitt.edu
#SBATCH --time=0-00:01:00
#SBATCH --ntasks-per-node=1

module purge
module load python/anaconda3.9-2021.11
module load cuda/11.5.0
conda activate neuraldecoding

python decoding-2.py $1 $2
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch