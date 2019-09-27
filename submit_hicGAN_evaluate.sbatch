#!/bin/bash

#SBATCH --job-name=run_hicGAN_evaluate
#SBATCH --qos gpu_access
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00
#SBATCH --mem=5GB
#SBATCH --output=/proj/yunligrp/users/minzhi/output_error/deephp/%x_%A_%a.out
#SBATCH --error=/proj/yunligrp/users/minzhi/output_error/deephp/%x_%A_%a.err
#SBATCH --mail-type=END,ALL
#SBATCH --mail-user=minzhi.hpc.status@gmail.com
#SBATCH --array=16

module purge
source activate hicgan_py27

chr=${SLURM_ARRAY_TASK_ID}
test_size=4
cd /pine/scr/m/i/minzhi/deephp/hicgan/
#python hicgan_data_split.py mouse ${chr} > result_data_split_${SLURM_JOB_ID}_${chr}.out 2>&1
#python run_hicGAN_mouse.py 0 checkpoint/${chr}-${test_size} graph/${chr}-${test_size} mouse ${chr} ${test_size} > result_run_hicgan_${SLURM_JOB_ID}_${chr}-${test_size}.out 2>&1
python hicGAN_evaluate.py 0 checkpoint_gpu/${chr}-${test_size} mouse ${chr} ${test_size} > result_evaluate_${SLURM_JOB_ID}_${chr}-${test_size}.out 2>&1

exit
