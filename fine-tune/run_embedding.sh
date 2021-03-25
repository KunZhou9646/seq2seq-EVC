#!/bin/bash

#SBATCH -w hlt02
#SBATCH --gres=gpu:1
#SBATCH --job-name=inference
#SBATCH -o res_emb.out 

source /home/zhoukun/miniconda3/bin/activate s2s

python inference_embedding.py -c '/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir_emotion_update/checkpoint_3200' --hparams speaker_A='Neutral',speaker_B='Happy',speaker_C='Sad',speaker_D='Angry',speaker_E='Surprise',training_list='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/emotion_list/testing_mel_list.txt',SC_kernel_size=1
