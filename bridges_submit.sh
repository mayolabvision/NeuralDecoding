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

module purge
module load anaconda3/2022.10
conda activate neuraldecoding

find ~/.ipython/profile_job* -maxdepth 0 -type d -ctime +1 | xargs rm -r
profile=job_${SLURM_JOB_ID}
echo "Creating profile_${profile}"
$HOME/conda/bin/ipython profile create ${profile}

$HOME/conda/bin/ipcontroller --ip="*" --profile=${profile} &
sleep 10

srun $HOME/conda/bin/ipengine --profile=${profile} --location=$(hostname) &
sleep 25

echo "Launching job for line $1"
python decoding-2.py $1 -p ${profile}

#cp files you'd like to move off of scratch
#mv files that you'd like moved off of scratch
