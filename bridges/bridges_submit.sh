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

on-conda 
source activate /jet/home/knoneman/miniconda3/envs/neuraldecoding

cd /jet/home/knoneman/NeuralDecoding 

python ../decoding-1.py $1
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
