import h5py
import torch
from tqdm import tqdm

# 전처리된 데이터를 HDF5 파일로 저장
def save_to_h5(data, h5_file, max_len):
    with h5py.File(h5_file, 'w') as f:
        mel_spectrograms = f.create_dataset('mel_spectrograms', (len(data), 80, max_len), dtype='float32', chunks=True)
        transcripts = f.create_dataset('transcripts', (len(data),), dtype=h5py.special_dtype(vlen=str))

        for idx, item in enumerate(tqdm(data, desc="Saving to HDF5", unit="sample")):
            mel = item['mel_spectrogram'].numpy()  # 이미 CPU에 있음
            mel_spectrograms[idx, :, :mel.shape[1]] = mel  # 고정된 최대 길이로 패딩
            transcripts[idx] = item['transcript']

if __name__ == '__main__':
    processed_data, max_len = torch.load('processed_data.pt')
    save_to_h5(processed_data, 'processed_data.h5', max_len)
    print("Processed data has been saved to HDF5.")
