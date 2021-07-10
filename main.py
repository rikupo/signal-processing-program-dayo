import numpy as np
import matplotlib.pyplot as plt

#音声処理用ライブラリ
import librosa
import librosa.display

# ICA
from sklearn.decomposition import FastICA

def main():

    audio_path1 = "s1.wav"
    audio_path2 = "s2.wav"
    audio1, sr1 = librosa.load(audio_path1)
    audio2, sr2 = librosa.load(audio_path2)

#ICAで音声分離 https://deepblue-ts.co.jp/voice-processing/independent_components_analysis/
#音声処理ライブラリ https://qiita.com/lilacs/items/a331a8933ec135f63ab1
#評価指標：dB 小さければよい？

def audio_evaluation(original,result):
    calc_sdr(original,result)
    calc_sir(original,result)
    calc_isr(original,result)
    pass

# Signal-to-distortion ratio (SDR): 出力音の歪みの少なさを評価する尺度. 値が大きいほど音声の分離性能が優れていることを示す．
def calc_sdr(original,result):
    pass

#Source to Interference Ratio（SIR）： 音声対目的音声以外の音声による歪比
def calc_sir(original,result):
    pass

# Source Image to SpatialdistortionRatio （ISR ）：音声対線形歪比
def calc_isr(original,result):
    pass


if __name__ == '__main__':
    main()
