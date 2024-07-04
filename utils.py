import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import random
import torch.nn as nn
import torchaudio
import numpy as np

# 문자 집합 정의
hangul_chars = [chr(i) for i in range(ord('가'), ord('힣') + 1)]
allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?~ " + ''.join(hangul_chars)

characters = allowed_characters
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for idx, char in enumerate(characters)}

class SpeechDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mel_spectrogram = self.data[idx]['mel_spectrogram']
        transcript = self.data[idx]['transcript']
        transcript_indices = torch.tensor([char_to_index[char] for char in transcript if char in char_to_index], dtype=torch.long)
        return mel_spectrogram, transcript_indices

def pad_collate_fn(batch):
    mel_spectrograms = [item[0] for item in batch]
    transcripts = [item[1] for item in batch]
    mel_spectrograms_padded = pad_sequence(mel_spectrograms, batch_first=True)
    transcripts_padded = pad_sequence(transcripts, batch_first=True)
    return mel_spectrograms_padded, transcripts_padded

def create_data_loader(data, batch_size):
    dataset = SpeechDataset(data)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn)

def split_data(data, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    total_size = len(data)
    train_size = int(total_size * train_ratio)
    val_size = int(total_size * val_ratio)
    test_size = total_size - train_size - val_size

    random.shuffle(data)
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]

    return train_data, val_data, test_data

class SpeechToTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeechToTextModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

def load_pcm(file_path, channels=1, sample_rate=16000, dtype=np.int16):
    with open(file_path, 'rb') as f:
        pcm_data = np.frombuffer(f.read(), dtype=dtype)
        pcm_data = pcm_data.astype(np.float32) / np.iinfo(dtype).max
        if channels > 1:
            pcm_data = pcm_data.reshape(-1, channels)
    return pcm_data, sample_rate

def predict(model, audio_path, device='cpu'):
    waveform, sr = load_pcm(audio_path)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80, n_fft=400)(torch.tensor(waveform).unsqueeze(0)).to(device)
    
    # mel_spectrogram이 (1, 80, L) 형태가 됨 -> (C, L) -> (1, C, L) -> (L, 1, C)
    mel_spectrogram = mel_spectrogram.squeeze(0).unsqueeze(0).permute(2, 0, 1)
    
    with torch.no_grad():
        outputs = model(mel_spectrogram)
    outputs = outputs.argmax(dim=2).squeeze(1)
    decoded_transcript = ''.join([index_to_char.get(idx.item(), '') for idx in outputs])
    return decoded_transcript