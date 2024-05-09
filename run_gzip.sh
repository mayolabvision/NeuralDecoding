#!/bin/bash
for FILE in \*.fastq; do
    echo ${FILE}
    sbatch -n 1 -t 1-00:00 --wrap="gzip ${FILE}"
    sleep 1 # pause to be kind to the scheduler
done
