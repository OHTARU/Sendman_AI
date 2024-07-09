import os
import sys
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Let_process_data import extract_features, load_pcm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import math

# Add parent directory to the module path
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# Define allowed characters including Hangul
hangul_chars = [chr(i) for i in range(ord('가'), ord('힣') + 1)]
allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?`~ " + ''.join(hangul_chars)

characters = allowed_characters
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for idx, char in enumerate(characters)}

class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, nhead=8, num_layers=6):
        super(TransformerModel, self).__init__()
        self.pos_encoder = PositionalEncoding(hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.encoder = nn.Linear(input_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, output_dim)

    def forward(self, src):
        src = self.encoder(src)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        return output

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class SpeechDataset(Dataset):
    def __init__(self, h5_files, transform=None):
        self.h5_files = h5_files
        self.transform = transform
        self.dataset_indices = []
        self.datasets = []
        self.lengths = []
        self.total_length = 0

        for h5_file in h5_files:
            with h5py.File(h5_file, 'r') as f:
                length = len(f['mel_spectrograms'])
                self.lengths.append(length)
                self.total_length += length
                self.datasets.append(h5_file)
                self.dataset_indices.extend([len(self.datasets) - 1] * length)

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        dataset_idx = self.dataset_indices[idx]
        internal_idx = idx - sum(self.lengths[:dataset_idx])

        with h5py.File(self.datasets[dataset_idx], 'r') as f:
            mel_spectrogram = torch.tensor(f['mel_spectrograms'][internal_idx])
            transcript = f['transcripts'][internal_idx].decode('utf-8')

        if self.transform:
            mel_spectrogram = self.transform(mel_spectrogram)
        
        filtered_transcript = ''.join([c for c in transcript if c in allowed_characters])
        transcript_indices = torch.tensor([char_to_index[c] for c in filtered_transcript])
        return mel_spectrogram, transcript_indices

def time_stretch(spec, rate=1.0):
    if not torch.is_complex(spec):
        spec = torch.stft(spec, n_fft=400, hop_length=128, window=torch.hamming_window(400), return_complex=True)
    stretched = torchaudio.transforms.TimeStretch(n_freq=spec.shape[1], hop_length=128)(spec, rate)
    return torch.istft(stretched, n_fft=400, hop_length=128, window=torch.hamming_window(400))

def freq_mask(spec, max_freqs=20):
    return torchaudio.transforms.FrequencyMasking(freq_mask_param=max_freqs)(spec.real)

def time_mask(spec, max_time=50):
    return torchaudio.transforms.TimeMasking(time_mask_param=max_time)(spec.real)

def augment_spectrogram(spec):
    if torch.is_complex(spec):
        spec = spec.abs()
    if np.random.rand() < 0.5:
        spec = time_stretch(spec, rate=np.random.uniform(0.8, 1.2))
    if np.random.rand() < 0.5:
        spec = freq_mask(spec)
    if np.random.rand() < 0.5:
        spec = time_mask(spec)
    return spec

def pad_collate_fn(batch):
    mel_spectrograms, transcripts = zip(*batch)
    max_len_mel = max([mel.size(1) for mel in mel_spectrograms])
    mel_spectrograms_padded = torch.stack([torch.nn.functional.pad(mel, (0, max_len_mel - mel.size(1))) for mel in mel_spectrograms])
    max_len_transcript = max([len(trans) for trans in transcripts])
    transcripts_padded = torch.stack([torch.nn.functional.pad(trans, (0, max_len_transcript - len(trans)), value=char_to_index.get(' ', 0)) for trans in transcripts])
    input_lengths = torch.tensor([mel.size(1) for mel in mel_spectrograms], dtype=torch.long)
    target_lengths = torch.tensor([len(trans) for trans in transcripts], dtype=torch.long)
    return mel_spectrograms_padded, transcripts_padded, input_lengths, target_lengths

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for mel_spectrograms, transcripts, input_lengths, target_lengths in dataloader:
            mel_spectrograms, transcripts, input_lengths, target_lengths = mel_spectrograms.to(device), transcripts.to(device), input_lengths.to(device), target_lengths.to(device)
            mel_spectrograms = mel_spectrograms.permute(0, 2, 1)  # (batch_size, sequence_len, feature_dim)로 변환
            outputs = model(mel_spectrograms)
            log_probs = outputs.log_softmax(2).permute(1, 0, 2)  # (sequence_len, batch_size, num_classes)로 변환
            loss = criterion(log_probs, transcripts, input_lengths, target_lengths)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def save_checkpoint(state, filename=r'D:\AI\checkpoint.pth.tar'):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    if os.path.isfile(checkpoint):
        print(f"Loading checkpoint '{checkpoint}'")
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch']
        best_loss = checkpoint['best_loss']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"Loaded checkpoint '{checkpoint}' (epoch {start_epoch})")
        return start_epoch, best_loss
    else:
        print(f"No checkpoint found at '{checkpoint}'")
        return 0, float('inf')

def train_model(model, train_loader, val_loader, num_epochs, lr=0.001, device='cpu', patience=5, checkpoint_file=r'D:\AI\checkpoint.pth.tar'):
    criterion = nn.CTCLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=2)
    scaler = GradScaler()
    
    start_epoch, best_val_loss = load_checkpoint(checkpoint_file, model, optimizer)
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for mel_spectrograms, transcripts, input_lengths, target_lengths in progress_bar:
            mel_spectrograms, transcripts, input_lengths, target_lengths = mel_spectrograms.to(device), transcripts.to(device), input_lengths.to(device), target_lengths.to(device)
            mel_spectrograms = mel_spectrograms.permute(0, 2, 1)  # (batch_size, sequence_len, feature_dim)로 변환
            
            optimizer.zero_grad()
            
            with autocast():
                outputs = model(mel_spectrograms)
                # CTCLoss는 Half precision을 지원하지 않으므로 autocast에서 제외
                log_probs = outputs.log_softmax(2).permute(1, 0, 2)  # (sequence_len, batch_size, num_classes)로 변환
                loss = criterion(log_probs, transcripts, input_lengths, target_lengths)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_epoch_loss}")
        
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")
        
        scheduler.step(val_loss)

        # Checkpoint 저장
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_val_loss,
            'optimizer': optimizer.state_dict(),
        }, filename=checkpoint_file)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), r'D:\\AI\\best_model.pth')
            print(f"Best model saved with validation loss: {val_loss}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print(f"Early stopping after {patience} epochs without improvement.")
                break

def predict(model, audio_path, device='cpu'):
    waveform, sr = load_pcm(audio_path)
    mel_spectrogram = extract_features(waveform, sr)
    mel_spectrogram = mel_spectrogram.unsqueeze(0).permute(2, 0, 1).to(device)
    with torch.no_grad():
        outputs = model(mel_spectrogram)
    outputs = outputs.argmax(dim=2).squeeze(1)
    decoded_transcript = ''.join([index_to_char.get(idx, '') for idx in outputs])
    return decoded_transcript

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = 80
    hidden_dim = 512
    output_dim = len(characters)
    h5_files = [
        os.path.join(r'D:\\AI\\output_dir', f'processed_data_part_{i}.h5') 
        for i in range(4)
    ]

    train_files, temp_files = train_test_split(h5_files, test_size=0.5, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=0.5, random_state=42)

    transform = augment_spectrogram if torch.cuda.is_available() else None
    
    train_dataset = SpeechDataset(train_files, transform=transform)
    val_dataset = SpeechDataset(val_files)
    test_dataset = SpeechDataset(test_files)

    # 배치 크기를 줄임 (기존 8에서 4로 줄임)
    train_loader = DataLoader(
        train_dataset,
        batch_size=4, shuffle=True, collate_fn=pad_collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=4, shuffle=False, collate_fn=pad_collate_fn, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=4, shuffle=False, collate_fn=pad_collate_fn, num_workers=4, pin_memory=True
    )

    model = TransformerModel(input_dim, hidden_dim, output_dim).to(device)
    if torch.cuda.device_count() > 1:W
        model = nn.DataParallel(model)

    print("Starting training...")
    train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, device=device, patience=10)

    test_loss = evaluate_model(model, test_loader, nn.CTCLoss().to(device), device)
    print(f"Test Loss: {test_loss}")

    audio_path = r"D:\AI\한국어 음성\평가용_데이터\eval_clean\KsponSpeech_E00001.pcm"
    transcript = predict(model, audio_path, device=device)
    print(f"Predicted transcript: {transcript}")

if __name__ == "__main__":
    main()
