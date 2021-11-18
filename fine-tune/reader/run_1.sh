#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=extract_mel
#SBATCH -o res_1.out 


source /home/zhoukun/miniconda3/bin/activate s2s

python generate_list.py
