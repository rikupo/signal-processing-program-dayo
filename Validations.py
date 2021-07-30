import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

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
        sep_audio_path2 = "Sep-signal1.wav"  # jupyter Separated
        sep_audio_path3 = "Sep-signal2.wav"  # Pipe Separated
        sep_audio_path1 = "Sep-signal3.wav"  # Mario64 Separated
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

    calc_CC = False
    if calc_CC:
        calc_CC_with_3_signals(audio1, audio2, audio3, "mix")
        calc_CC_with_3_signals(original_audio1, original_audio2,original_audio3 , "ori")
        calc_CC_with_3_signals(sep_audio1, sep_audio2, sep_audio3, "seq")

    calc_additonal_evaluation = True
    if calc_additonal_evaluation:
        originals = np.concatenate([[original_audio1],[original_audio2],[original_audio3]])

        print("SDR,SIR,SAR")
        originals = originals.transpose()

        print(audio_evaluation2(originals,sep_audio1,0))
        print(audio_evaluation2(originals, sep_audio2, 1))
        print(audio_evaluation2(originals, sep_audio3, 2))

        print(audio_evaluation(original_audio1,sep_audio1))
        print(audio_evaluation(original_audio2, sep_audio2))
        print(audio_evaluation(original_audio3, sep_audio3))

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

# SNR/SDR: https://github.com/JusperLee/Calculate-SNR-SDR
#評価指標：dB 小さければよい？
# https://hal.inria.fr/inria-00544230/file/vincent_TASLP06bis.pdf

# Modified from The sigsep that is Open Resources for Audio Source Separation https://github.com/sigsep/bsseval/issues/3
def audio_evaluation(reference_signals, estimated_signal, scaling=True):
    Rss = np.dot(reference_signals.transpose(), reference_signals)
    this_s = reference_signals

    if scaling:
        a = np.dot(this_s, estimated_signal) / Rss
    else:
        a = 1
    # print(f"Debug:: {a}")
    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true ** 2).sum()
    Snn = (e_res ** 2).sum()

    SDR = 10 * math.log10(Sss / Snn)

    # Get the SIR
    Rsr = np.dot(reference_signals.transpose(), e_res)
    # linalg.solve arg1: A of AX = B arg2: B return: X
    # b = np.linalg.solve(Rss, Rsr)
    b = Rsr/Rss
    e_interf = np.dot(reference_signals, b)
    e_artif = e_res - e_interf

    SIR = 10 * math.log10(Sss / (e_interf ** 2).sum())
    SAR = 10 * math.log10(Sss / (e_artif ** 2).sum())

    return SDR, SIR, SAR


def audio_evaluation2(reference_signals, estimated_signal, j, scaling=True):
    Rss = np.dot(reference_signals.transpose(), reference_signals)
    this_s = reference_signals[:, j]

    if scaling:
        # get the scaling factor for clean sources
        a = np.dot(this_s, estimated_signal) / Rss[j, j]
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true ** 2).sum()
    Snn = (e_res ** 2).sum()

    SDR = 10 * math.log10(Sss / Snn)

    # Get the SIR
    Rsr = np.dot(reference_signals.transpose(), e_res)
    b = np.linalg.solve(Rss, Rsr)

    e_interf = np.dot(reference_signals, b)
    e_artif = e_res - e_interf

    SIR = 10 * math.log10(Sss / (e_interf ** 2).sum())
    SAR = 10 * math.log10(Sss / (e_artif ** 2).sum())

    return SDR, SIR, SAR

def calc_CC_with_3_signals(signal1,signal2,signal3,title = "Given"):
    signal1 = pd.Series(signal1)
    signal2 = pd.Series(signal2)
    signal3 = pd.Series(signal3)
    Correlation_coefficient12 = signal1.corr(signal2)
    Correlation_coefficient13 = signal1.corr(signal3)
    Correlation_coefficient23 = signal2.corr(signal3)
    print(f"CC of {title} signal: {Correlation_coefficient12}, {Correlation_coefficient13}, {Correlation_coefficient23}")

if __name__ == '__main__':
    main()

# SIR/SDR式 https://ipsj.ixsq.nii.ac.jp/ej/?action=repository_uri&item_id=113127&file_id=1&file_no=1