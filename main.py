import numpy as np

#音声処理用ライブラリ
import librosa
import librosa.display
import soundfile as sf

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
    if (sr1 != sr2) or (sr1 != sr3) or (sr2 != sr3):
        print("Wrong Sampling Rate")
        sys.exit(1)

    Run_ICA = True
    if Run_ICA:
        print("ICA process start")
        X = np.concatenate((audio1.reshape(-1, 1), audio2.reshape(-1, 1), audio3.reshape(-1,1)), 1)
        transformer = FastICA(n_components=3, random_state=0)
        X_t = transformer.fit_transform(X)
        out1 = X_t[:,0]
        out2 = X_t[:,1]
        out3 = X_t[:,2]
        print("ICA process end")
        print(f"out put shape {X_t.shape}")

        signal1 = np.array(out1)
        signal2 = np.array(out2)
        signal3 = np.array(out3)

        save_output = True
        if save_output:
            sf.write("Sep-signal1.wav", signal1, sr1)
            sf.write("Sep-signal2.wav", signal2, sr1)
            sf.write("Sep-signal3.wav", signal3, sr1)

if __name__ == '__main__':
    main()
