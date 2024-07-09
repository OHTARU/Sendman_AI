import torch
import os
# # 간단한 텐서를 생성하여 GPU로 전송하고 연산을 수행
# x = torch.tensor([1.0, 2.0, 3.0]).to('cuda')
# y = x * 2
# print(y)
# print(y.device)

# import torch
# import matplotlib.pyplot as plt

# # 전처리된 데이터 로드
# processed_data, max_len = torch.load('D:\\AI\\processed_data1.pt')

# # 데이터 샘플 시각화
# sample = processed_data[0]  # 첫 번째 샘플
# mel_spectrogram = sample['mel_spectrogram']

# plt.figure(figsize=(10, 4))
# plt.imshow(mel_spectrogram, aspect='auto', origin='lower')
# plt.title('Mel-Spectrogram')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.colorbar()
# plt.show()
import h5py

def check_hdf5_file(file_path):
    with h5py.File(file_path, 'r') as f:
        mel_spectrograms = f['mel_spectrograms'][:]
        transcripts = f['transcripts'][:]
        print(f"File: {file_path}")
        print(f"  mel_spectrograms shape: {mel_spectrograms.shape}")
        print(f"  transcripts length: {len(transcripts)}")
        print(f"  File size: {os.path.getsize(file_path) / (1024 * 1024)} MB")

# Check original file
check_hdf5_file(r'D:\AI\processed_data1.h5')

# Check split files
for i in range(10):
    check_hdf5_file(rf'D:\AI\output_dir\processed_data_part_{i}.h5')
