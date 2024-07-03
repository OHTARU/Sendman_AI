import pickle

# 피클 파일 로드
with open('processed_data.pkl', 'rb') as f:
    loaded_data = pickle.load(f)

# 데이터 확인
print(f"Number of entries: {len(loaded_data)}")
print(f"Shape of first mel_spectrogram: {loaded_data[0]['mel_spectrogram'].shape}")
print(f"First transcript: {loaded_data[0]['transcript']}")
