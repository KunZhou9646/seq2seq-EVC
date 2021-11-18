#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=inference
#SBATCH -o res_2.out 

source /home/zhoukun/miniconda3/bin/activate new

python inference_A.py -c '/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train-ser-smi-acc_update_ser/outdir_emotion_update_final/checkpoint_1800' --num 20 --hparams validation_list='/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/reader/emotion_list/evaluation_mel_list.txt',SC_kernel_size=1

#python inference.py -c /home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir_emotion_03_22/checkpoint_4500 --num 20 --hparams validation_list='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/emotion_list/evaluation_mel_list.txt',SC_kernel_size=1
