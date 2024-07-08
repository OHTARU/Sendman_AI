import os
import torch
import torch.nn as nn
import torchaudio
import numpy as np

# Define allowed characters including Hangul
hangul_chars = [chr(i) for i in range(ord('가'), ord('힣') + 1)]
allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?`~ " + ''.join(hangul_chars)

# Ensure the length of allowed_characters is exactly 11267
if len(allowed_characters) < 11267:
    for i in range(11267 - len(allowed_characters)):
        allowed_characters += chr(1000 + i)
elif len(allowed_characters) > 11267:
    allowed_characters = allowed_characters[:11267]

characters = allowed_characters
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for idx, char in enumerate(characters)}

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

def extract_features(waveform, sr, n_mels=80, n_fft=400):
    waveform = torch.tensor(waveform).unsqueeze(0)  # (1, N) 형태로 변환
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft)(waveform)
    return mel_spectrogram.squeeze(0)

def predict(model, audio_path, device='cpu'):
    waveform, sr = load_pcm(audio_path)
    mel_spectrogram = extract_features(waveform, sr)
    mel_spectrogram = mel_spectrogram.unsqueeze(0).permute(2, 0, 1).to(device)
    
    print(f"Waveform shape: {waveform.shape}, Sample rate: {sr}")
    print(f"Mel-spectrogram shape: {mel_spectrogram.shape[1:]}")

    with torch.no_grad():
        outputs = model(mel_spectrogram)
    
    print(f"Model outputs shape: {outputs.shape}")
    outputs = outputs.argmax(dim=2).squeeze(1)
    print(f"Outputs after argmax and squeeze: {outputs}")
    
    decoded_transcript = ''.join([index_to_char.get(idx.item(), '') for idx in outputs])
    return decoded_transcript

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = 80
    hidden_dim = 128  # 모델 학습 시 사용한 값과 일치하도록 수정
    output_dim = len(characters)

    model = SpeechToTextModel(input_dim, hidden_dim, output_dim).to(device)

    # 모델 로드
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    # 예제 오디오 파일 경로
    audio_path = r"D:\AI\한국어 음성\한국어_음성_분야\KsponSpeech_05\KsponSpeech_05\KsponSpeech_0497\KsponSpeech_496001.pcm"

    # 예측
    transcript = predict(model, audio_path, device=device)
    print(f"Predicted transcript: {transcript}")
