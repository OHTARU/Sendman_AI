import numpy as np
from scipy.io import wavfile

# .wav 파일 로드
original_wav_file_path = 'SPK003KBSEC032F002.wav'
processed_file_path = 'SPK003KBSEC032F002_features.npy'

# .wav 파일 로드
sample_rate, original_data = wavfile.read(original_wav_file_path)

# 전처리된 .npy 파일 로드
processed_data = np.load(processed_file_path)

# 데이터 구조 확인
print("Original Data Shape:", original_data.shape)
print("Processed Data Shape:", processed_data.shape)

# 통계 비교
original_mean = np.mean(original_data)
processed_mean = np.mean(processed_data)
original_std = np.std(original_data)
processed_std = np.std(processed_data)

print("Original Mean:", original_mean)
print("Processed Mean:", processed_mean)
print("Original Std:", original_std)
print("Processed Std:", processed_std)

# 데이터 샘플 비교
sample_size = 10  # 비교할 데이터 샘플의 크기
original_sample = original_data[:sample_size]
processed_sample = processed_data[:sample_size]

print("Original Sample:\n", original_sample)
print("Processed Sample:\n", processed_sample)

# 데이터 손실 여부 확인 (일치 여부 확인)
data_equal = np.array_equal(original_data, processed_data)
print("Data is identical:", data_equal)
