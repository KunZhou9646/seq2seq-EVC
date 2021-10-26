# seq2seq-EVC
This is the implementation of our Interspeech 2021 paper "Limited data emotional voice conversion leveraging text-to-speech: two-stage sequence-to-sequence training".

## Implementation Details

The text encoder is 3-layer of 1D CNN with a kernel size of 5 and the channel of 512, followed by 1-layer of  256-cell BLSTM and a fully connected (FC) layer with the output channel of 512. The seq2seq ASR encoder consists of an encoder which is a  2-layer 256-cell BLSTM, and a decoder which is a  1-layer 512-cell LSTM with an attention layer and followed by a FC layer with the output channel of 512. The style encoder is 2-layer of 128-cell BLSTM followed by a FC layer with the output channel of 128. The classifier is 4-layer of FC with the channel of \{512, 512, 512, 99\}. The seq2seq decoder follows the same model architecture of the one used in Tacotron [1]. During pre-training, we set the learning rate as 0.0001 for 200 epochs. For the emotion adaptation, we set the learning rate as 0.0001 and half it every 7 epochs. We set the batch size as 64 and 32 for pre-training and adaptation respectively. The WaveRNN vocoder predicts 9-bits waveform with mu-law companding. Its implementation follows a publicly available version: https://github.com/fatchord/WaveRNN



**Note:** 
The codes are based on Non-parallel Speaker Voice Conversion: https://github.com/jxzhanggg/nonparaSeq2seqVC_code

[1]  Y.  Wang,  R.  Skerry-Ryan,  D.  Stanton,  Y.  Wu,  R.  J.  Weiss, N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengioet al., “Tacotron:Towards  end-to-end  speech  synthesis,” Proc.  Interspeech  2017,pp. 4006–4010, 2017.
## Database:
We use ESD database, which is an emotional speech database that can be downloaded here: https://hltsingapore.github.io/ESD/. To run the codes, you first need to customize your data path correctly, and generate phoneme transcriptions with Festival. More details can be found in https://github.com/jxzhanggg/nonparaSeq2seqVC_code.
## To run the codes:

**1. Installation**
```Bash
$ pip install -r requirements.txt
```
**2. Stage I: Style Pre-training**

The pre-training procedure is same as the pretraining in  https://github.com/jxzhanggg/nonparaSeq2seqVC_code. You can download the pre-trained models from Stage I: Style Initialization here: https://drive.google.com/file/d/1oqk-PSREwpFNTyeREwcUry13WZ1LYl6U/view?usp=sharing. With the released pre-trained models, you can directly perform Stage II: Emotion Training. If you would like to pre-train it by yourself, you can try the following:
```Bash
$ python train.py -l logdir \
-o outdir --n_gpus=1 --hparams=speaker_adversial_loss_w=20.,ce_loss=False,speaker_classifier_loss_w=0.1,contrastive_loss_w=30.
```

**3. Stage II: Emotion Training**
```Bash
$ python train.py -l logdir \
-o outdir_emotion_IS --n_gpus=1 -c '/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir/checkpoint_234000 (The path to your Pre-trained models from Stage I)' --warm_start
```
**3. Run-time Inference**
(1) Generate emotion embedding from the emotion encoder:
```Bash
$ python inference_embedding.py -c '/home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir_emotion_update/checkpoint_3200' --hparams speaker_A='Neutral',speaker_B='Happy',speaker_C='Sad',speaker_D='Angry',speaker_E='Surprise',training_list='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/emotion_list/testing_mel_list.txt',SC_kernel_size=1
```
(2) Convert the source speech to the target emotion:
```Bash
$ python inference.py -c /home/zhoukun/nonparaSeq2seqVC_code-master/pre-train/outdir_emotion_update/checkpoint_3200 --num 20 --hparams validation_list='/home/zhoukun/nonparaSeq2seqVC_code-master/fine-tune/reader/emotion_list/evaluation_mel_list.txt',SC_kernel_size=1
```
Please customize inference.py to generate your intended emotion type.
