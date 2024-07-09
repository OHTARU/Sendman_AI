import torch
import h5py
from utils import SpeechToTextModel, create_data_loader, split_data
import torch.nn as nn

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 문자 집합 정의
hangul_chars = [chr(i) for i in range(ord('가'), ord('힣') + 1)]
allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?~ " + ''.join(hangul_chars)

characters = allowed_characters
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for idx, char in enumerate(characters)}

# 모델 초기화
input_dim = 80  # Mel-spectrogram feature size
hidden_dim = 256
output_dim = len(characters)  # 문자 집합의 크기

model = SpeechToTextModel(input_dim, hidden_dim, output_dim).to(device)

# 모델 로드
model.load_state_dict(torch.load('model_epoch_10.pth'))
model.eval()

# 전처리된 데이터를 로드합니다.
h5_file = 'processed_data.h5'
with h5py.File(h5_file, 'r') as f:
    mel_spectrograms = f['mel_spectrograms'][:]
    transcripts = f['transcripts'][:]

# 데이터 분할
processed_data = [{'mel_spectrogram': torch.tensor(mel).to(device), 'transcript': trans.decode('utf-8')} for mel, trans in zip(mel_spectrograms, transcripts)]
_, val_data, _ = split_data(processed_data)

# DataLoader 생성
val_loader = create_data_loader(val_data, batch_size=16)

# 모델 검증
model.eval()
val_loss = 0
criterion = nn.CTCLoss()
with torch.no_grad():
    for mel_spectrograms, transcripts in val_loader:
        mel_spectrograms, transcripts = mel_spectrograms.to(device), transcripts.to(device)
        mel_spectrograms = mel_spectrograms.permute(2, 0, 1)  # (N, C, L) -> (L, N, C)
        target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long).to(device)
        input_lengths = torch.full(size=(mel_spectrograms.size(1),), fill_value=mel_spectrograms.size(0), dtype=torch.long).to(device)
        
        outputs = model(mel_spectrograms)
        loss = criterion(outputs, transcripts, input_lengths, target_lengths)
        val_loss += loss.item()

avg_val_loss = val_loss / len(val_loader)
print(f"Validation Loss: {avg_val_loss}")
