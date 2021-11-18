#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=extract
#SBATCH -o res.out 


source /home/zhoukun/miniconda3/bin/activate s2s

python extract_features.py
