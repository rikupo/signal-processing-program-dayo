import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#音声処理用ライブラリ
import IPython.display
from IPython.display import display
import librosa
import librosa.display

import sys

# ICA
from sklearn.decomposition import FastICA

def main():

    audio_path1 = "x1.wav"
    audio_path2 = "x2.wav"
    audio_path3 = "x3.wav"
    original_audio_path1 = "s1.wav"
    original_audio_path2 = "s2.wav"
    original_audio_path3 = "s3.wav"

    try:
        audio1, sr1 = librosa.load(audio_path1)
        audio2, sr2 = librosa.load(audio_path2)
        audio3, sr3 = librosa.load(audio_path3)
        original_audio1, original_sr1 = librosa.load(original_audio_path1)
        original_audio2, original_sr2 = librosa.load(original_audio_path2)
        original_audio3, original_sr3 = librosa.load(original_audio_path3)
    except FileNotFoundError:
        print("open file failed")
        sys.exit(1)
    signal1 = pd.Series(audio1)
    signal2 = pd.Series(audio2)
    signal3 = pd.Series(audio3)
    Correlation_coefficient12 = signal1.corr(signal2)
    Correlation_coefficient13 = signal1.corr(signal3)
    Correlation_coefficient23 = signal2.corr(signal3)
    print(f"CC of original signal: {Correlation_coefficient12}, {Correlation_coefficient13}, {Correlation_coefficient23}")

    plt.figure(figsize=(20, 7))

    plt.subplot(3, 3, 2)
    plt.plot(audio1, color="red")
    plt.ylabel("Amplitude")

    plt.subplot(3, 3, 4)
    plt.plot(original_audio1, color="red")
    plt.ylabel("Amplitude")

    plt.subplot(3, 3, 5)
    plt.plot(original_audio2, color="green")
    plt.ylabel("Amplitude")

    plt.subplot(3, 3, 6)
    plt.plot(original_audio3, color="blue")
    plt.ylabel("Amplitude")

    plt.xlabel("time")
    #plt.show()

    if 1:
        print("ICA process start")
        X = np.concatenate((audio1.reshape(-1, 1), audio2.reshape(-1, 1), audio3.reshape(-1,1)), 1)
        transformer = FastICA(n_components=3, random_state=0)
        X_t = transformer.fit_transform(X)
        out1 = X_t[:,0]
        out2 = X_t[:,1]
        out3 = X_t[:,2]
        print("ICA process end")
        print(f"out put shape {X_t.shape}")

        signal1 = pd.Series(np.array(out1))
        signal2 = pd.Series(np.array(out2))
        signal3 = pd.Series(np.array(out3))
        # pd.series(out1)みたいな感じだとout1自体がndarray型のオブジェクトだから動かない．must be 1-dementionalって．

        Correlation_coefficient12 = signal1.corr(signal2)
        Correlation_coefficient13 = signal1.corr(signal3)
        Correlation_coefficient23 = signal2.corr(signal3)

        print(f"CC after process: {Correlation_coefficient12}, {Correlation_coefficient13}, {Correlation_coefficient23}")

        #plt.figure(figsize=(20, 5))

        plt.subplot(3, 3, 8)
        plt.plot(X_t[:, 0],color = "red")
        plt.ylabel("Amplitude")

        plt.subplot(3, 3, 9)
        plt.plot(X_t[:, 1],color = "green")
        plt.ylabel("Amplitude")

        plt.subplot(3, 3, 7)
        plt.plot(X_t[:, 2],color = "blue")
        plt.ylabel("Amplitude")

        plt.xlabel("time")
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
