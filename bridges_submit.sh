#!/bin/bash -l
#SBATCH -p RM-shared
#SBATCH -J neuraldecoding
#SBATCH -o runs/outfiles/out.o%j
#SBATCH -N 1
#SBATCH -n 5
#SBATCH --mail-type=all
#SBATCH --mail-user=knoneman@andrew.cmu.edu
#SBATCH -t 00:01:00
#SBATCH --ntasks-per-node=8

module purge
module load anaconda3/2022.10
module load cuda/11.4
conda activate neuraldecoding

python decoding-2.py $1 0
python decoding-2.py $1 1
python decoding-2.py $1 2
python decoding-2.py $1 3
python decoding-2.py $1 4
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
