#!/bin/bash -l
#SBATCH -p batch
#SBATCH -J C60
#SBATCH -o outfiles/C60.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=all
#SBATCH --mail-user=kendranoneman@u.boisestate.edu
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1

source activate py3.9.5

python decoding-1.py $1
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
