#!/bin/bash -l
#SBATCH -p RM-shared
#SBATCH -J neuraldecoding
#SBATCH -o runs/outfiles/out.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=all
#SBATCH --mail-user=knoneman@andrew.cmu.edu
#SBATCH -t 05:00:00
#SBATCH --ntasks-per-node=8

module purge
module load anaconda3/2022.10
module load cuda/11.4
conda activate neuraldecoding

python decoding-2.py $1 $2
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
