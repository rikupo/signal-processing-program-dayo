import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#音声処理用ライブラリ
import IPython.display
from IPython.display import display
import librosa
import librosa.display
import soundfile as sf

import sys

# ICA
from sklearn.decomposition import FastICA

def main():
    open_data = True
    if open_data:
        audio_path1 = "x1.wav"
        audio_path2 = "x2.wav"
        audio_path3 = "x3.wav"
        original_audio_path1 = "s1.wav"  # Mario64
        original_audio_path2 = "s2.wav"  # jupyter
        original_audio_path3 = "s3.wav"  # Pipe
        sep_audio_path2 = "Sep-signal1.wav"  # jupyter
        sep_audio_path3 = "Sep-signal2.wav"  # Pipe
        sep_audio_path1 = "Sep-signal3.wav"  # Mario64
        try:
            audio1, sr1 = librosa.load(audio_path1)
            audio2, sr2 = librosa.load(audio_path2)
            audio3, sr3 = librosa.load(audio_path3)
            original_audio1, original_sr1 = librosa.load(original_audio_path1)
            original_audio2, original_sr2 = librosa.load(original_audio_path2)
            original_audio3, original_sr3 = librosa.load(original_audio_path3)
            sep_audio1, sep_sr1 = librosa.load(sep_audio_path1)
            sep_audio2, sep_sr2 = librosa.load(sep_audio_path2)
            sep_audio3, sep_sr3 = librosa.load(sep_audio_path3)
        except FileNotFoundError:
            print("open file failed")
            sys.exit(1)
        if (sr1 != sr2) or (sr1 != sr3) or (sr2 != sr3):
            print("Wrong Sampling Rate")
            sys.exit(1)

    signal1 = pd.Series(audio1)
    signal2 = pd.Series(audio2)
    signal3 = pd.Series(audio3)
    Correlation_coefficient12 = signal1.corr(signal2)
    Correlation_coefficient13 = signal1.corr(signal3)
    Correlation_coefficient23 = signal2.corr(signal3)
    print(f"CC of original signal: {Correlation_coefficient12}, {Correlation_coefficient13}, {Correlation_coefficient23}")

    signal1 = pd.Series(sep_audio1)
    signal2 = pd.Series(sep_audio2)
    signal3 = pd.Series(sep_audio3)
    Correlation_coefficient12 = signal1.corr(signal2)
    Correlation_coefficient13 = signal1.corr(signal3)
    Correlation_coefficient23 = signal2.corr(signal3)
    print(f"CC of sep signal: {Correlation_coefficient12}, {Correlation_coefficient13}, {Correlation_coefficient23}")

    show_graph = False
    if show_graph:
        plt.figure(figsize=(20, 7))

        plt.subplot(3, 3, 1)
        plt.plot(audio1, color="gray")


        plt.title('Mixed Signal 1')

        plt.subplot(3, 3, 2)
        plt.plot(audio2, color="gray")

        plt.title('Mixed Signal 2')

        plt.subplot(3, 3, 3)
        plt.plot(audio3, color="gray")

        plt.title('Mixed Signal 3')

        plt.subplot(3, 3, 4)
        plt.plot(original_audio1, color="#e8dcba")

        plt.title('Original Signal 1')


        plt.subplot(3, 3, 5)
        plt.plot(original_audio2, color="#a3cca1")
        plt.title('Original Signal 2')


        plt.subplot(3, 3, 6)
        plt.plot(original_audio3, color="#9fc5cc")

        plt.title('Original Signal 3')


        plt.subplot(3, 3, 7)
        plt.plot(sep_audio1,color = "#e8dcba")

        plt.title('Separated Signal 1')


        plt.subplot(3, 3, 8)
        plt.plot(sep_audio2,color = "#a3cca1")

        plt.title('Separated Signal 2')


        plt.subplot(3, 3, 9)
        plt.plot(sep_audio3,color = "#9fc5cc")

        plt.title('Separated Signal 3')

        plt.tight_layout()

        plt.show()

#ICAで音声分離 https://deepblue-ts.co.jp/voice-processing/independent_components_analysis/
#音声処理ライブラリ https://qiita.com/lilacs/items/a331a8933ec135f63ab1
# SNR/SDR: https://github.com/JusperLee/Calculate-SNR-SDR
#評価指標：dB 小さければよい？

def audio_evaluation(original,result):
    calc_sdr(original,result)
    calc_sir(original,result)
    calc_isr(original,result)
    pass

# Signal-to-distortion ratio (SDR): 出力音の歪みの少なさを評価する尺度. 値が大きいほど音声の分離性能が優れていることを示す．
def calc_sdr(original,result):
    pass

# Source to Interference Ratio（SIR）： 音声対目的音声以外の音声による歪比
def calc_sir(original,result):
    pass

# Source Image to SpatialdistortionRatio （ISR ）：音声対線形歪比
def calc_isr(original,result):
    pass

if __name__ == '__main__':
    main()
