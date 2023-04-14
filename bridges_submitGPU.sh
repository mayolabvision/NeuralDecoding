#!/bin/bash -l
#SBATCH -p GPU-shared
#SBATCH -J neuraldecoding
#SBATCH -o runs/outfiles/out.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=all
#SBATCH --mail-user=knoneman@andrew.cmu.edu
#SBATCH -t 01:00:00
#SBATCH --gpus=v100-32:4

module purge
module load anaconda3/2022.10
conda activate neuraldecoding-gpu

python decoding-1.py $1
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
