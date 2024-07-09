import h5py
import os
import numpy as np
from tqdm import tqdm

def split_hdf5_file(input_file, output_dir, num_splits, compression=None):
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with h5py.File(input_file, 'r') as f:
        mel_spectrograms = f['mel_spectrograms'][:]
        transcripts = f['transcripts'][:]
    
    num_samples = len(mel_spectrograms)
    split_size = num_samples // num_splits
    
    for i in tqdm(range(num_splits), desc="Splitting HDF5 file"):
        start_idx = i * split_size
        end_idx = (i + 1) * split_size if i != num_splits - 1 else num_samples
        
        output_file = os.path.join(output_dir, f'processed_data_part_{i}.h5')
        with h5py.File(output_file, 'w') as out_f:
            out_f.create_dataset('mel_spectrograms', data=mel_spectrograms[start_idx:end_idx], compression=compression)
            out_f.create_dataset('transcripts', data=transcripts[start_idx:end_idx], compression=compression)

# Usage example
split_hdf5_file(r'D:\AI\processed_data1.h5', r'D:\AI\output_dir', num_splits=10, compression='gzip')
