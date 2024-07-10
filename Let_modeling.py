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
module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

# Define allowed characters including Hangul
hangul_chars = [chr(i) for i in range(ord("가"), ord("힣") + 1)]
allowed_characters = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?`~ "
    + "".join(hangul_chars)
)

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


class SpeechDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file

    def __len__(self):
        with h5py.File(self.h5_file, "r") as f:
            return len(f["mel_spectrograms"])

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, "r") as f:
            mel_spectrogram = torch.tensor(f["mel_spectrograms"][idx])
            transcript = f["transcripts"][idx].decode("utf-8")
        filtered_transcript = "".join(
            [c for c in transcript if c in allowed_characters]
        )
        transcript_indices = torch.tensor(
            [char_to_index[c] for c in filtered_transcript]
        )
        return mel_spectrogram, transcript_indices


def split_dataset(h5_file, test_size=0.2, val_size=0.1):
    with h5py.File(h5_file, "r") as f:
        indices = list(range(len(f["mel_spectrograms"])))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=42
    )
    train_indices, val_indices = train_test_split(
        train_indices, test_size=val_size, random_state=42
    )
    return train_indices, val_indices, test_indices


def create_subset_loader(h5_file, indices, batch_size, num_workers=2):
    dataset = SpeechDataset(h5_file)
    subset = torch.utils.data.Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=pad_collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )


def pad_collate_fn(batch):
    mel_spectrograms, transcripts = zip(*batch)
    max_len_mel = max([mel.size(1) for mel in mel_spectrograms])
    mel_spectrograms_padded = torch.stack(
        [
            torch.nn.functional.pad(mel, (0, max_len_mel - mel.size(1)))
            for mel in mel_spectrograms
        ]
    )
    max_len_transcript = max([len(trans) for trans in transcripts])
    transcripts_padded = torch.stack(
        [
            torch.nn.functional.pad(
                trans,
                (0, max_len_transcript - len(trans)),
                value=char_to_index.get(" ", 0),
            )
            for trans in transcripts
        ]
    )
    return mel_spectrograms_padded, transcripts_padded


def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for mel_spectrograms, transcripts in dataloader:
            mel_spectrograms, transcripts = mel_spectrograms.to(device), transcripts.to(
                device
            )
            mel_spectrograms = mel_spectrograms.permute(2, 0, 1)
            target_lengths = torch.tensor(
                [len(t) for t in transcripts], dtype=torch.long
            ).to(device)
            input_lengths = torch.full(
                size=(mel_spectrograms.size(1),),
                fill_value=mel_spectrograms.size(0),
                dtype=torch.long,
            ).to(device)
            outputs = model(mel_spectrograms)
            loss = criterion(outputs, transcripts, input_lengths, target_lengths)
            total_loss += loss.item()
    return total_loss / len(dataloader)


def train_model(model, train_loader, val_loader, num_epochs=10, lr=0.001, device="cpu"):
    criterion = nn.CTCLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    best
