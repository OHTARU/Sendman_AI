import numpy as np

features = np.load(r"D:\138.뉴스 대본 및 앵커 음성 데이터\새 폴더\SPK003\SPK003KBSEC032\SPK003KBSEC032F001_features.npy")
print("Shape of features:", features.shape)
print("First chunk MFCC shape:", features[0].shape)
print("First chunk MFCC values:\n", features[0])

import matplotlib.pyplot as plt
import librosa.display

first_chunk_mfcc = features[0]
plt.figure(figsize=(10, 4))
librosa.display.specshow(first_chunk_mfcc, sr=16000, x_axis='time')
plt.colorbar()
plt.title('MFCC')
plt.tight_layout()
plt.show()