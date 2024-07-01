import librosa
import noisereduce as nr
import os
import numpy as np
from tqdm import tqdm
import gc

def load_audio(file_path, target_sr=16000):
    try:
        audio, sr = librosa.load(file_path, sr=target_sr)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None, None
    return audio, sr

def reduce_noise(audio, sr):
    return nr.reduce_noise(y=audio, sr=sr)

def normalize_audio(audio):
    return librosa.util.normalize(audio)

def split_audio_on_silence(audio, sr, silence_threshold=30, chunk_size=2048):
    intervals = librosa.effects.split(audio, top_db=silence_threshold, frame_length=chunk_size)
    return [audio[start:end] for start, end in intervals]

def extract_features(audio, sr, n_mfcc=13):
    n_fft = min(2048, len(audio))
    hop_length = n_fft // 2
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    return mfccs

def preprocess_audio(file_path, target_sr=16000, silence_threshold=30, chunk_size=2048, n_mfcc=13, max_len=100):
    audio, sr = load_audio(file_path, target_sr)
    if audio is None:
        return None
    
    audio = reduce_noise(audio, sr)
    chunks = split_audio_on_silence(audio, sr, silence_threshold, chunk_size)
    
    features = []
    for chunk in chunks:
        normalized_chunk = normalize_audio(chunk)
        mfccs = extract_features(normalized_chunk, sr, n_mfcc)
        if mfccs.shape[1] < max_len:
            pad_width = max(0, max_len - mfccs.shape[1])
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :max_len]
        
        features.append(mfccs)
    
    if not features:
        return None
    
    return np.stack(features, axis=0)

def preprocess_all_audios_in_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    total_files = sum([len(files) for _, _, files in os.walk(folder_path) if any(file.endswith('.wav') for file in files)])
    
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_file_path = os.path.join(root, file)
                    try:
                        print(f'Processing file: {audio_file_path}')
                        features = preprocess_audio(audio_file_path)
                        if features is None:
                            print(f"Failed to process {audio_file_path}")
                            pbar.update(1)
                            continue
                        
                        output_file_path = os.path.join(output_folder, os.path.relpath(audio_file_path, folder_path)).replace('.wav', '_features.npy')
                        output_dir = os.path.dirname(output_file_path)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        
                        np.save(output_file_path, features)
                    except Exception as e:
                        print(f"Error processing {audio_file_path}: {e}")
                    finally:
                        pbar.update(1)
                        del features
                        gc.collect()

audio_directory = r'D:\138.뉴스 대본 및 앵커 음성 데이터\01-1.정식개방데이터\Training\01.원천데이터\TS'
output_directory = r'D:\138.뉴스 대본 및 앵커 음성 데이터\MFCC'
preprocess_all_audios_in_folder(audio_directory, output_directory)
