#!/bin/bash -l
#SBATCH -p RM-shared
#SBATCH -J neuraldecoding
#SBATCH -o outfiles/out.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=all
#SBATCH --mail-user=knoneman@andrew.cmu.edu
#SBATCH -t 00:01:00
#SBATCH --ntasks-per-node=8

module purge
module load anaconda3/2022.10
conda activate neuraldecoding

module load /jet/home/knoneman/NeuralDecoding/handy_functions

python ../decoding-1.py $1
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
