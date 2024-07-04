import os
import torch
import torchaudio
from utils import SpeechToTextModel, predict

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 모델 초기화
input_dim = 80  # Mel-spectrogram feature size
hidden_dim = 256
output_dim = len(predict.__globals__['characters'])  # 문자 집합의 크기

model = SpeechToTextModel(input_dim, hidden_dim, output_dim).to(device)

# 모델 로드 (절대 경로 사용)
model_path = r'processed_data.h5'
model.load_state_dict(torch.load(model_path))
model.eval()

# 테스트할 오디오 파일 경로
audio_path = r"D:\한국어 음성\한국어_음성_분야\KsponSpeech_03\KsponSpeech_03\KsponSpeech_0249\KsponSpeech_248029.pcm"

# 오디오 파일을 로드하고 예측
waveform, sr = torchaudio.load(audio_path)
mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_mels=80, n_fft=400)(waveform).to(device)
mel_spectrogram = mel_spectrogram.unsqueeze(0).permute(2, 0, 1)  # (C, L) -> (1, C, L) -> (L, 1, C)

predicted_transcript = predict(model, mel_spectrogram)
print(f"Predicted transcript: {predicted_transcript}")
