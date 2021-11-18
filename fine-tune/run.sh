#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --job-name=train
#SBATCH -o res.out


source /home/zhoukun/miniconda3/bin/activate s2s


# you can set the hparams by using --hparams=xxx


python train.py -l logdir \
-o outdir_emotion_IS --n_gpus=1 -c '/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir/checkpoint_234000' --warm_start

