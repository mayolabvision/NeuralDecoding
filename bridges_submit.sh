#!/bin/bash -l
#SBATCH -p RM-shared
#SBATCH -J neuraldecoding
#SBATCH -o runs/outfiles/out.o%j
#SBATCH -N 1
#SBATCH --mail-type=all
#SBATCH --mail-user=knoneman@andrew.cmu.edu
#SBATCH -t 00:30:00
#SBATCH --ntasks-per-node=1

module purge
module load anaconda3/2022.10
conda activate neuraldecoding

python decoding-2.py $1 $2
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
