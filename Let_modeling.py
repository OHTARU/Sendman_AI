import os
import sys
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import DataLoader, Dataset
import h5py
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import itertools
from Let_process_data import extract_features, load_pcm

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

class SpeechToTextModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SpeechToTextModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=3, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x

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
        filtered_transcript = ''.join([c for c in transcript if c in allowed_characters])
        transcript_indices = torch.tensor([char_to_index[c] for c in filtered_transcript])
        return mel_spectrogram, transcript_indices

def split_dataset(h5_file, test_size=0.2, val_size=0.1):
    with h5py.File(h5_file, 'r') as f:
        indices = list(range(len(f['mel_spectrograms'])))
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=val_size, random_state=42)
    return train_indices, val_indices, test_indices

def create_subset_loader(h5_file, indices, batch_size, num_workers=0):
    dataset = SpeechDataset(h5_file)
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=True, collate_fn=pad_collate_fn, num_workers=num_workers, pin_memory=True)

def pad_collate_fn(batch):
    mel_spectrograms, transcripts = zip(*batch)
    max_len_mel = max([mel.size(1) for mel in mel_spectrograms])
    mel_spectrograms_padded = torch.stack([torch.nn.functional.pad(mel, (0, max_len_mel - mel.size(1))) for mel in mel_spectrograms])
    max_len_transcript = max([len(trans) for trans in transcripts])
    transcripts_padded = torch.stack([torch.nn.functional.pad(trans, (0, max_len_transcript - len(trans)), value=char_to_index.get(' ', 0)) for trans in transcripts])
    return mel_spectrograms_padded, transcripts_padded

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for mel_spectrograms, transcripts in dataloader:
            mel_spectrograms, transcripts = mel_spectrograms.to(device), transcripts.to(device)
            mel_spectrograms = mel_spectrograms.permute(2, 0, 1)
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long).to(device)
            input_lengths = torch.full(size=(mel_spectrograms.size(1),), fill_value=mel_spectrograms.size(0), dtype=torch.long).to(device)
            outputs = model(mel_spectrograms)
            loss = criterion(outputs, transcripts, input_lengths, target_lengths)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, num_epochs, lr=0.001, device='cpu'):
    criterion = nn.CTCLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for mel_spectrograms, transcripts in progress_bar:
            mel_spectrograms, transcripts = mel_spectrograms.to(device), transcripts.to(device)
            mel_spectrograms = mel_spectrograms.permute(2, 0, 1)
            target_lengths = torch.tensor([len(t) for t in transcripts], dtype=torch.long).to(device)
            input_lengths = torch.full(size=(mel_spectrograms.size(1),), fill_value=mel_spectrograms.size(0), dtype=torch.long).to(device)
            optimizer.zero_grad()
            outputs = model(mel_spectrograms)
            loss = criterion(outputs, transcripts, input_lengths, target_lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Training Loss: {avg_epoch_loss}")
        
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with validation loss: {val_loss}")

def grid_search(h5_file, param_grid, num_epochs, train_indices, val_indices, device='cpu'):
    best_params = None
    best_val_loss = float('inf')
    all_params = [dict(zip(param_grid, v)) for v in itertools.product(*param_grid.values())]
    for params in all_params:
        print(f"Evaluating parameters: {params}")
        model = SpeechToTextModel(input_dim=80, hidden_dim=params['hidden_dim'], output_dim=len(characters)).to(device)
        train_loader = create_subset_loader(h5_file, train_indices, batch_size=params['batch_size'], num_workers=2)
        val_loader = create_subset_loader(h5_file, val_indices, batch_size=params['batch_size'], num_workers=2)
        train_model(model, train_loader, val_loader, num_epochs, lr=params['lr'], device=device)
        val_loss = evaluate_model(model, val_loader, nn.CTCLoss().to(device), device)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = params
            torch.save(model.state_dict(), 'best_model.pth')
            print(f"Best model saved with validation loss: {val_loss}")
    return best_params

def predict(model, audio_path, device='cpu'):
    waveform, sr = load_pcm(audio_path)
    mel_spectrogram = extract_features(waveform, sr)
    mel_spectrogram = mel_spectrogram.unsqueeze(0).permute(2, 0, 1).to(device)
    with torch.no_grad():
        outputs = model(mel_spectrogram)
    outputs = outputs.argmax(dim=2).squeeze(1)
    decoded_transcript = ''.join([index_to_char.get(idx, '') for idx in outputs])
    return decoded_transcript

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = 80
    hidden_dim = 256
    output_dim = len(characters)
    h5_file = 'processed_data.h5'

    train_indices, val_indices, test_indices = split_dataset(h5_file)
    param_grid = {
        'hidden_dim': [128, 256, 512],
        'batch_size': [4, 8, 16],
        'lr': [0.001, 0.0001]
    }
    
    best_params = grid_search(h5_file, param_grid, num_epochs=10, train_indices=train_indices, val_indices=val_indices, device=device)
    print(f"Best hyperparameters: {best_params}")

    best_model = SpeechToTextModel(input_dim=80, hidden_dim=best_params['hidden_dim'], output_dim=len(characters)).to(device)
    best_model.load_state_dict(torch.load('best_model.pth'))

    train_loader = create_subset_loader(h5_file, train_indices, batch_size=best_params['batch_size'], num_workers=2)
    val_loader = create_subset_loader(h5_file, val_indices, batch_size=best_params['batch_size'], num_workers=2)
    test_loader = create_subset_loader(h5_file, test_indices, batch_size=best_params['batch_size'], num_workers=2)
    
    train_model(best_model, train_loader, val_loader, num_epochs=10, lr=best_params['lr'], device=device)

    test_loss = evaluate_model(best_model, test_loader, nn.CTCLoss().to(device), device)
    print(f"Test Loss: {test_loss}")

    audio_path = r"D:\한국어 음성\한국어_음성_분야\KsponSpeech_03\KsponSpeech_03\KsponSpeech_0249\KsponSpeech_248029.pcm"
    transcript = predict(best_model, audio_path, device=device)
    print(f"Predicted transcript: {transcript}")
