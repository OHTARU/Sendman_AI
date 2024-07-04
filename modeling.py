import os
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py
from tqdm import tqdm  # tqdm 라이브러리 추가

# 모든 한글 문자 포함
hangul_chars = [chr(i) for i in range(ord('가'), ord('힣') + 1)]
allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?`~ " + ''.join(hangul_chars)

characters = allowed_characters
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for idx, char in enumerate(characters)}

# STT 모델 정의
class SpeechToTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeechToTextModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

# .pcm 파일 로드 및 전처리
def load_pcm(file_path, channels=1, sample_rate=16000, dtype=np.int16):
    with open(file_path, 'rb') as f:
        pcm_data = np.frombuffer(f.read(), dtype=dtype)
        pcm_data = pcm_data.astype(np.float32) / np.iinfo(dtype).max
        if channels > 1:
            pcm_data = pcm_data.reshape(-1, channels)
    return pcm_data, sample_rate

# 특성 추출 (Mel-spectrogram)
def extract_features(waveform, sr, n_mels=80, n_fft=400):
    waveform = torch.tensor(waveform).unsqueeze(0)  # (1, N) 형태로 변환
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft)(waveform)
    return mel_spectrogram.squeeze(0)

# 학습 데이터 로드 및 준비 (전처리된 데이터를 사용)
class SpeechDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file

    def __len__(self):
        with h5py.File(self.h5_file, 'r') as f:
            return len(f['mel_spectrograms'])

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as f:
            mel_spectrogram = torch.tensor(f['mel_spectrograms'][idx])
            transcript = f['transcripts'][idx].decode('utf-8')
        # 허용된 문자로만 필터링
        filtered_transcript = ''.join([char for char in transcript if char in allowed_characters])
        transcript_indices = torch.tensor([char_to_index[char] for char in filtered_transcript])
        return mel_spectrogram, transcript_indices

def pad_collate_fn(batch):
    mel_spectrograms, transcripts = zip(*batch)
    
    # 멜-스펙트로그램의 패딩
    max_len_mel = max([mel.size(1) for mel in mel_spectrograms])
    mel_spectrograms_padded = torch.stack([torch.nn.functional.pad(mel, (0, max_len_mel - mel.size(1))) for mel in mel_spectrograms])

    # 전사의 패딩
    max_len_transcript = max([len(trans) for trans in transcripts])
    transcripts_padded = torch.stack([torch.nn.functional.pad(trans, (0, max_len_transcript - len(trans)), value=char_to_index.get(' ', 0)) for trans in transcripts])

    return mel_spectrograms_padded, transcripts_padded

def create_data_loader(h5_file, batch_size, num_workers=0):
    dataset = SpeechDataset(h5_file)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=num_workers)

# 모델 학습 함수
def train_model(model, dataloader, num_epochs, lr=0.001):
    criterion = nn.CTCLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0  # 에포크별 손실 초기화
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for mel_spectrograms, transcripts in progress_bar:
            mel_spectrograms = mel_spectrograms.permute(2, 0, 1)  # (N, C, L) -> (L, N, C)
            target_lengths = torch.tensor([len(t) for t in transcripts])
            input_lengths = torch.full(size=(mel_spectrograms.size(1),), fill_value=mel_spectrograms.size(0), dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(mel_spectrograms)
            loss = criterion(outputs, transcripts, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()  # 배치 손실을 에포크 손실에 추가
            progress_bar.set_postfix(loss=loss.item())
        avg_epoch_loss = epoch_loss / len(dataloader)  # 에포크 손실의 평균 계산
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_epoch_loss}")

# 추론 함수
def predict(model, audio_path):
    waveform, sr = load_pcm(audio_path)
    mel_spectrogram = extract_features(waveform, sr)
    mel_spectrogram = mel_spectrogram.unsqueeze(0).permute(2, 0, 1)  # (C, L) -> (1, C, L) -> (L, 1, C)
    with torch.no_grad():
        outputs = model(mel_spectrogram)
    outputs = outputs.argmax(dim=2).squeeze(1)
    decoded_transcript = ''.join([index_to_char.get(idx, '') for idx in outputs])
    return decoded_transcript

# 모델 초기화
input_dim = 80  # Mel-spectrogram feature size
hidden_dim = 256
output_dim = len(characters)  # 문자 집합의 크기

model = SpeechToTextModel(input_dim, hidden_dim, output_dim)

# 전처리된 데이터를 로드합니다.
h5_file = 'processed_data.h5'

dataloader = create_data_loader(h5_file, batch_size=4, num_workers=2)  # 배치 크기를 4로 줄이고 num_workers 설정

# 모델 학습
train_model(model, dataloader, num_epochs=10)

# 예제 사용
audio_path = r"D:\한국어 음성\한국어_음성_분야\KsponSpeech_03\KsponSpeech_03\KsponSpeech_0249\KsponSpeech_248029.pcm"
transcript = predict(model, audio_path)
print(f"Predicted transcript: {transcript}")
