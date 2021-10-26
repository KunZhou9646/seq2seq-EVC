# seq2seq-EVC
This is the implementation of our Interspeech 2021 paper "Limited data emotional voice conversion leveraging text-to-speech: two-stage sequence-to-sequence training".

## Implementation Details

The text encoder is 3-layer of 1D CNN with a kernel size of 5 and the channel of 512, followed by 1-layer of  256-cell BLSTM and a fully connected (FC) layer with the output channel of 512. The seq2seq ASR encoder consists of an encoder which is a  2-layer 256-cell BLSTM, and a decoder which is a  1-layer 512-cell LSTM with an attention layer and followed by a FC layer with the output channel of 512. The style encoder is 2-layer of 128-cell BLSTM followed by a FC layer with the output channel of 128. The classifier is 4-layer of FC with the channel of \{512, 512, 512, 99\}. The seq2seq decoder follows the same model architecture of the one used in Tacotron [1]. During pre-training, we set the learning rate as 0.0001 for 200 epochs. For the emotion adaptation, we set the learning rate as 0.0001 and half it every 7 epochs. We set the batch size as 64 and 32 for pre-training and adaptation respectively. The WaveRNN vocoder predicts 9-bits waveform with mu-law companding. Its implementation follows a publicly available version: https://github.com/fatchord/WaveRNN



**Note:** 
The codes are based on Non-parallel Speaker Voice Conversion: https://github.com/jxzhanggg/nonparaSeq2seqVC_code

[1]  Y.  Wang,  R.  Skerry-Ryan,  D.  Stanton,  Y.  Wu,  R.  J.  Weiss, N. Jaitly, Z. Yang, Y. Xiao, Z. Chen, S. Bengioet al., “Tacotron:Towards  end-to-end  speech  synthesis,” Proc.  Interspeech  2017,pp. 4006–4010, 2017.

## To run the codes:

**1. Stage I: Style Pre-training**

The pre-training procedure is same as the pretraining in  https://github.com/jxzhanggg/nonparaSeq2seqVC_code. You can download the pre-trained models from Stage I: Style Initialization here: https://drive.google.com/file/d/1oqk-PSREwpFNTyeREwcUry13WZ1LYl6U/view?usp=sharing. With the released pre-trained models, you can directly perform Stage II: Emotion Training.
