#!/bin/bash

#SBATCH -w hlt06
#SBATCH --gres=gpu:1
#SBATCH --job-name=inference
#SBATCH -o res_2.out 

source /home/zhoukun/miniconda3/bin/activate s2s

python inference.py -c /home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir_emotion_update/checkpoint_3200 --num 20 --hparams validation_list='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/emotion_list/evaluation_mel_list.txt',SC_kernel_size=1
