#!/bin/bash -l
#SBATCH -p batch
#SBATCH -J neuraldecoding
#SBATCH -o runs/outfiles/out.o%j
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mail-type=all
#SBATCH --mail-user=kendranoneman@u.boisestate.edu
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1

module purge
source activate py3.10.8
module load numpy

python ../decoding-1.py $1
#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
