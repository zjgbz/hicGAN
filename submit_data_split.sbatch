#!/bin/bash

#SBATCH --job-name=data_split
##SBATCH --qos gpu_access
##SBATCH --partition=gpu
##SBATCH --gres=gpu:1
#SBATCH --partition=general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --mem=60GB
#SBATCH --output=/proj/yunligrp/users/minzhi/output_error/deephp/%x_%A_%a.out
#SBATCH --error=/proj/yunligrp/users/minzhi/output_error/deephp/%x_%A_%a.err
#SBATCH --mail-type=END,ALL
#SBATCH --mail-user=minzhi.hpc.status@gmail.com
#SBATCH --array=11

module purge
source activate hicgan_py27

chr=${SLURM_ARRAY_TASK_ID}
test_size=4
cd /pine/scr/m/i/minzhi/deephp/hicgan/
python hicgan_data_split.py mouse ${chr} ${test_size} > result_data_split_${SLURM_JOB_ID}_${chr}.out 2>&1
#python run_hicGAN_mouse.py 0 checkpoint/${chr} graph/${chr} mouse ${chr} > result_run_hicgan_${SLURM_JOB_ID}_${chr}.out 2>&1
#python hicGAN_evaluate.py 0 checkpoint/${chr} mouse ${chr} > result_evaluate_${SLURM_JOB_ID}_${chr}.out 2>&1

exit
