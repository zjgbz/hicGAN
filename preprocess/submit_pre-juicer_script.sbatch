#!/bin/bash

#SBATCH --job-name=hicgan_pre-preprocess
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=20:00:00
#SBATCH --mem=10GB
#SBATCH --output=/proj/yunligrp/users/minzhi/output_error/deephp/%x_%A_%a.out
#SBATCH --error=/proj/yunligrp/users/minzhi/output_error/deephp/%x_%A_%a.err
#SBATCH --mail-type=END,ALL
#SBATCH --mail-user=minzhi.hpc.status@gmail.com # send-to address

module purge
module load cuda/8.0
#module load juicer/1.6.2
#conda activate hicgan

cell_type="GM12878"
resolution="10000"
juicer_path="../juicer_tools.jar"

bash preprocess_1.sh ${cell_type} ${resolution} ${juicer_path} > result_${SLURM_JOB_NAME}_${SLURM_JOB_ID}.out 2>&1

#conda deactivate

exit
