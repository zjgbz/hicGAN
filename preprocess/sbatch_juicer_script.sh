#!/bin/bash

#SBATCH --job-name=hicgan_preprocess
#SBATCH -p general
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00
#SBATCH --mem=55GB
#SBATCH --output=/proj/yunligrp/users/minzhi/output_error/deephp/%x_%A_%a.out
#SBATCH --error=/proj/yunligrp/users/minzhi/output_error/deephp/%x_%A_%a.err
#SBATCH --mail-type=END,ALL
#SBATCH --mail-user=minzhi.hpc.status@gmail.com # send-to address

ratio=16
chrom=$1
DPATH=$2
CELL=$3
resolution=$4
juicer_tool=$5
java -jar $juicer_tool dump observed NONE $DPATH/$2/total_merged.hic $1 $1 BP $resolution $DPATH/$2/intra_NONE/chr$1_10k_intra_NONE.txt 
java -jar $juicer_tool dump observed VC $DPATH/$2/total_merged.hic $1 $1 BP $resolution $DPATH/$2/intra_VC/chr$1_10k_intra_VC.txt 
java -jar $juicer_tool dump observed KR $DPATH/$2/total_merged.hic $1 $1 BP $resolution $DPATH/$2/intra_KR/chr$1_10k_intra_KR.txt 

java -jar $juicer_tool dump observed NONE $DPATH/$2/total_merged_downsample_ratio_$ratio.hic $1 $1 BP $resolution $DPATH/$2/intra_NONE/chr$1_10k_intra_NONE_downsample_ratio$ratio.txt
java -jar $juicer_tool dump observed VC $DPATH/$2/total_merged_downsample_ratio_$ratio.hic $1 $1 BP $resolution $DPATH/$2/intra_VC/chr$1_10k_intra_VC_downsample_ratio$ratio.txt
java -jar $juicer_tool dump observed KR $DPATH/$2/total_merged_downsample_ratio_$ratio.hic $1 $1 BP $resolution $DPATH/$2/intra_KR/chr$1_10k_intra_KR_downsample_ratio$ratio.txt

