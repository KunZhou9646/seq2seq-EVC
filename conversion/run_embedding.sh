#!/bin/bash


#SBATCH --gres=gpu:1
#SBATCH --job-name=inference
#SBATCH -o res_emb.out 

source /home/zhoukun/miniconda3/bin/activate new

python inference_embedding.py -c '/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train-ser-smi-acc_update_ser/outdir_emotion_update_final/checkpoint_1800' --hparams speaker_A='Angry',speaker_B='Happy',speaker_C='Sad',speaker_D='Neutral',training_list='/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/reader/emotion_list/testing_mel_list.txt',SC_kernel_size=1

#python inference_embedding.py -c '/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir_emotion_03_22/checkpoint_4500' --hparams speaker_A='Neutral',speaker_B='Happy',speaker_C='Sad',speaker_D='Angry',speaker_E='Surprise',training_list='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/emotion_list/testing_mel_list.txt',SC_kernel_size=1

