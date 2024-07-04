import os
import torch
import torchaudio
from utils import SpeechToTextModel, predict, load_pcm

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 문자 집합 정의
hangul_chars = [chr(i) for i in range(ord('가'), ord('힣') + 1)]
allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?`~ " + ''.join(hangul_chars)

characters = allowed_characters
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for idx, char in enumerate(characters)}

# 모델 초기화
input_dim = 80  # Mel-spectrogram feature size
hidden_dim = 256
output_dim = len(characters)  # 문자 집합의 크기

model = SpeechToTextModel(input_dim, hidden_dim, output_dim).to(device)

# 모델 로드 (절대 경로 사용)
model_path = r'model_epoch_10.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# 테스트할 오디오 파일 경로
audio_path = r"D:\한국어 음성\한국어_음성_분야\KsponSpeech_03\KsponSpeech_03\KsponSpeech_0249\KsponSpeech_248029.pcm"

# 오디오 파일을 로드하고 예측
predicted_transcript = predict(model, audio_path, device=device)
print(f"Predicted transcript: {predicted_transcript}")
