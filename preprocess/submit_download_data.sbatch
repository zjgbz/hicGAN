#!/bin/bash

#SBATCH --job-name=hicgan_download
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=55GB
#SBATCH --output=/proj/yunligrp/users/minzhi/output_error/deephp/%x_%A_%a.out
#SBATCH --error=/proj/yunligrp/users/minzhi/output_error/deephp/%x_%A_%a.err
#SBATCH --mail-type=END,ALL
#SBATCH --mail-user=minzhi.hpc.status@gmail.com # send-to address
#SBATCH --array=4,14-16

cell_type="GM12878"

bash raw_data_download_script.sh ${cell_type} ${SLURM_ARRAY_TASK_ID} > result_${SLURM_JOB_NAME}_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out 2>&1

exit
