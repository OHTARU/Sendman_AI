import h5py
import torch
from tqdm import tqdm
import os

def save_to_h5_split(data, output_dir, max_len, num_splits):
    os.makedirs(output_dir, exist_ok=True)
    split_size = len(data) // num_splits

    for i in range(num_splits):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_splits - 1 else len(data)
        output_file = os.path.join(output_dir, f'processed_data_part_{i}.h5')

        with h5py.File(output_file, 'w') as f:
            mel_spectrograms = f.create_dataset('mel_spectrograms', (end_idx - start_idx, 80, max_len), dtype='float32', chunks=(1, 80, max_len), compression='gzip')
            transcripts = f.create_dataset('transcripts', (end_idx - start_idx,), dtype=h5py.special_dtype(vlen=str))

            for idx, item in enumerate(tqdm(data[start_idx:end_idx], desc=f"Saving to {output_file}", unit="sample")):
                mel = item['mel_spectrogram'].numpy()
                mel_spectrograms[idx, :, :mel.shape[1]] = mel
                transcripts[idx] = item['transcript']

if __name__ == '__main__':
    processed_data, max_len = torch.load('D:\\AI\\processed_data1.pt')
    save_to_h5_split(processed_data, r'D:\\AI\\output_dir', max_len, num_splits=10)
    print("Processed data has been saved to HDF5 files.")
