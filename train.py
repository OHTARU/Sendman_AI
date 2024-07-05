import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm
from utils import SpeechToTextModel, create_data_loader, split_data

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

# 전처리된 데이터를 로드합니다.
h5_file = 'processed_data.h5'
with h5py.File(h5_file, 'r') as f:
    mel_spectrograms = f['mel_spectrograms'][:]
    transcripts = f['transcripts'][:]

# 데이터 분할
processed_data = [{'mel_spectrogram': torch.tensor(mel).to(device), 'transcript': trans.decode('utf-8')} for mel, trans in zip(mel_spectrograms, transcripts)]
train_data, val_data, test_data = split_data(processed_data)

# DataLoader 생성
train_loader = create_data_loader(train_data, batch_size=16)
val_loader = create_data_loader(val_data, batch_size=16)

# 모델 학습 함수
def train_model(model, train_loader, val_loader, num_epochs, lr=0.001):
    criterion = nn.CTCLoss()
    optimizer = Adam(model.parameters(), lr=lr)
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for mel_spectrograms, transcripts in progress_bar:
            mel_spectrograms, transcripts = mel_spectrograms.to(device), transcripts.to(device)
            mel_spectrograms = mel_spectrograms.permute(2, 0, 1)  # (N, C, L) -> (L, N, C)
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long).to(device)
            input_lengths = torch.full(size=(mel_spectrograms.size(1),), fill_value=mel_spectrograms.size(0), dtype=torch.long).to(device)

            optimizer.zero_grad()
            outputs = model(mel_spectrograms)
            loss = criterion(outputs, transcripts, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # 검증 데이터로 모델 평가
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for mel_spectrograms, transcripts in val_loader:
                mel_spectrograms, transcripts = mel_spectrograms.to(device), transcripts.to(device)
                mel_spectrograms = mel_spectrograms.permute(2, 0, 1)
                target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long).to(device)
                input_lengths = torch.full(size=(mel_spectrograms.size(1),), fill_value=mel_spectrograms.size(0), dtype=torch.long).to(device)
                
                outputs = model(mel_spectrograms)
                loss = criterion(outputs, transcripts, input_lengths, target_lengths)
                epoch_val_loss += loss.item()
        
        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss}, Val Loss: {avg_val_loss}")

        # 모델 저장
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

    # 학습 및 검증 손실 시각화
    plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.show()

# 모델 학습
train_model(model, train_loader, val_loader, num_epochs=10)
