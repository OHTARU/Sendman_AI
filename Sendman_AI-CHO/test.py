import os
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import h5py
from tqdm import tqdm
from Let_process_data import extract_features, load_pcm

# 필요한 모듈이 로드되는지 확인
try:
    from Let_process_data import extract_features, load_pcm

    print("모듈이 성공적으로 로드되었습니다.")
except ImportError as e:
    print(f"모듈 로드 오류: {e}")

# 상수 정의
h5_file = "D:\\AI\\processed_data.h5"
audio_path = (
    r"D:\AI\한국어 음성\한국어_음성_분야\KsponSpeech_02\KsponSpeech_02\KsponSpeech_0125"
)
num_epochs = 2  # 테스트 목적으로 에포크 수를 낮게 설정
test_size = 0.2
val_size = 0.1

# HDF5 파일이 존재하는지 확인
if not os.path.exists(h5_file):
    raise FileNotFoundError(
        f"{h5_file}을(를) 찾을 수 없습니다. 파일 경로를 확인하십시오."
    )

# 오디오 파일이 존재하는지 확인
if not os.path.exists(audio_path):
    raise FileNotFoundError(
        f"{audio_path}을(를) 찾을 수 없습니다. 파일 경로를 확인하십시오."
    )

# 데이터셋 분할
train_indices, val_indices, test_indices = split_dataset(
    h5_file, test_size=test_size, val_size=val_size
)

# 모델 초기화
input_dim = 80
hidden_dim = 256
output_dim = len(characters)

model = SpeechToTextModel(
    input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim
)
print(model)

# 데이터 로더 생성
batch_size = 4  # 테스트 목적으로 배치 크기를 낮게 설정

train_loader = create_subset_loader(
    h5_file, train_indices, batch_size=batch_size, num_workers=0
)
val_loader = create_subset_loader(
    h5_file, val_indices, batch_size=batch_size, num_workers=0
)
test_loader = create_subset_loader(
    h5_file, test_indices, batch_size=batch_size, num_workers=0
)

# 데이터 로더 출력 확인
print("데이터 로더 출력 확인:")
for mel_spectrograms, transcripts in train_loader:
    print("멜 스펙트로그램 크기:", mel_spectrograms.shape)
    print("전사 크기:", transcripts.shape)
    break

# 훈련 루프
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

train_model(
    model, train_loader, val_loader, num_epochs=num_epochs, lr=0.001, device=device
)

# 테스트 셋 평가
test_loss = evaluate_model(model, test_loader, nn.CTCLoss().to(device), device)
print(f"테스트 손실: {test_loss}")

# 샘플 오디오로 예측 수행
transcript = predict(model, audio_path, device=device)
print(f"예측된 전사: {transcript}")


# ----------------------------------------------------------------------------------------------


import h5py
import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# 멜 스펙트로그램과 전사 데이터를 HDF5 파일에 저장하는 함수
def process_and_save(idx, item, mel_spectrograms, transcripts, max_len):
    mel = item["mel_spectrogram"].numpy()
    mel_spectrograms[idx, :, : mel.shape[1]] = mel
    transcripts[idx] = item["transcript"]
    return idx


# 데이터를 HDF5 파일로 저장하는 함수
def save_to_h5(data, h5_file, max_len):
    with h5py.File(h5_file, "w") as f:
        # 멜 스펙트로그램 데이터셋 생성
        mel_spectrograms = f.create_dataset(
            "mel_spectrograms",
            (len(data), 80, max_len),
            dtype="float32",
            chunks=(1, 80, max_len),  # 청크 크기 설정
            compression="gzip",  # 압축 사용
        )
        # 전사 데이터셋 생성
        transcripts = f.create_dataset(
            "transcripts",
            (len(data),),
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",  # 압축 사용
        )

        # 스레드를 사용하여 데이터를 병렬로 저장
        with ThreadPoolExecutor(max_workers=4) as executor:  # 최대 4개의 스레드 사용
            futures = [
                executor.submit(
                    process_and_save, idx, item, mel_spectrograms, transcripts, max_len
                )
                for idx, item in enumerate(data)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(data),
                desc="Saving to HDF5",
                unit="sample",
            ):
                future.result()


# 테스트 데이터 생성 함수
def generate_test_data(num_samples, max_len):
    data = []
    for i in range(num_samples):
        mel_spectrogram = torch.rand(
            80, np.random.randint(1, max_len + 1)
        )  # 랜덤 멜 스펙트로그램 생성
        transcript = f"sample transcript {i}"  # 샘플 전사 생성
        data.append({"mel_spectrogram": mel_spectrogram, "transcript": transcript})
    return data, max_len


# 테스트 함수
def test_save_to_h5():
    # 테스트용 데이터 생성
    num_samples = 100  # 샘플 수
    max_len = 300  # 최대 길이
    data, max_len = generate_test_data(num_samples, max_len)

    # HDF5 파일로 저장
    h5_file = "test_processed_data.h5"
    save_to_h5(data, h5_file, max_len)

    # HDF5 파일 확인
    with h5py.File(h5_file, "r") as f:
        assert "mel_spectrograms" in f
        assert "transcripts" in f
        assert len(f["mel_spectrograms"]) == num_samples
        assert len(f["transcripts"]) == num_samples
        print("HDF5 파일이 올바르게 저장되었습니다.")


if __name__ == "__main__":
    test_save_to_h5()


# ----------------------------------------------------------------------------------------------


import os
import torch
import h5py
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def create_test_h5_file(file_path, num_samples=100, max_len=300):
    """테스트용 HDF5 파일을 생성합니다."""
    with h5py.File(file_path, "w") as f:
        mel_spectrograms = f.create_dataset(
            "mel_spectrograms", (num_samples, 80, max_len), dtype="float32"
        )
        transcripts = f.create_dataset(
            "transcripts", (num_samples,), dtype=h5py.special_dtype(vlen=str)
        )
        for i in range(num_samples):
            mel_spectrograms[i] = torch.rand(80, max_len).numpy()
            transcripts[i] = f"sample transcript {i}"


def test_split_dataset():
    h5_file = "test_data.h5"
    create_test_h5_file(h5_file)

    train_indices, val_indices, test_indices = split_dataset(
        h5_file, test_size=0.2, val_size=0.1
    )

    assert len(train_indices) > 0
    assert len(val_indices) > 0
    assert len(test_indices) > 0
    print("Dataset split test passed.")


def test_data_loader():
    h5_file = "test_data.h5"
    create_test_h5_file(h5_file)

    train_indices, val_indices, test_indices = split_dataset(
        h5_file, test_size=0.2, val_size=0.1
    )

    train_loader = create_subset_loader(h5_file, train_indices, batch_size=8)
    val_loader = create_subset_loader(h5_file, val_indices, batch_size=8)
    test_loader = create_subset_loader(h5_file, test_indices, batch_size=8)

    for mel_spectrograms, transcripts in train_loader:
        assert mel_spectrograms.shape[0] == 8
        assert transcripts.shape[0] == 8
        break

    print("Data loader test passed.")


def test_model_training():
    h5_file = "test_data.h5"
    create_test_h5_file(h5_file)

    train_indices, val_indices, _ = split_dataset(h5_file, test_size=0.2, val_size=0.1)

    train_loader = create_subset_loader(h5_file, train_indices, batch_size=8)
    val_loader = create_subset_loader(h5_file, val_indices, batch_size=8)

    input_dim = 80
    hidden_dim = 128
    output_dim = len(characters)

    model = SpeechToTextModel(input_dim, hidden_dim, output_dim).to("cpu")

    train_model(model, train_loader, val_loader, num_epochs=1, lr=0.001, device="cpu")

    print("Model training test passed.")


def test_model_evaluation():
    h5_file = "test_data.h5"
    create_test_h5_file(h5_file)

    train_indices, val_indices, test_indices = split_dataset(
        h5_file, test_size=0.2, val_size=0.1
    )

    test_loader = create_subset_loader(h5_file, test_indices, batch_size=8)

    input_dim = 80
    hidden_dim = 128
    output_dim = len(characters)

    model = SpeechToTextModel(input_dim, hidden_dim, output_dim).to("cpu")

    criterion = nn.CTCLoss().to("cpu")
    test_loss = evaluate_model(model, test_loader, criterion, device="cpu")

    assert test_loss >= 0
    print("Model evaluation test passed.")


def test_prediction():
    h5_file = "test_data.h5"
    create_test_h5_file(h5_file)

    input_dim = 80
    hidden_dim = 128
    output_dim = len(characters)

    model = SpeechToTextModel(input_dim, hidden_dim, output_dim).to("cpu")
    model.load_state_dict(torch.load("best_model.pth"))

    audio_path = "sample_audio.wav"

    # 샘플 오디오 파일 생성
    waveform = torch.rand(1, 16000)  # 1초 길이의 임의의 오디오 데이터 생성
    torchaudio.save(audio_path, waveform, 16000)

    transcript = predict(model, audio_path, device="cpu")

    assert isinstance(transcript, str)
    print("Prediction test passed.")


if __name__ == "__main__":
    print("Running tests...")
    test_split_dataset()
    test_data_loader()
    test_model_training()
    test_model_evaluation()
    test_prediction()
    print("All tests passed.")


# ----------------------------------------------------------------------------------------------


import os
import torch
import torchaudio
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader


# Helper function to create a test HDF5 file
def create_test_h5_file(file_path, num_samples=10, max_len=300):
    """테스트용 HDF5 파일을 생성합니다."""
    with h5py.File(file_path, "w") as f:
        mel_spectrograms = f.create_dataset(
            "mel_spectrograms", (num_samples, 80, max_len), dtype="float32"
        )
        transcripts = f.create_dataset(
            "transcripts", (num_samples,), dtype=h5py.special_dtype(vlen=str)
        )
        for i in range(num_samples):
            mel_spectrograms[i] = torch.rand(80, max_len).numpy()
            transcripts[i] = f"sample transcript {i}"


# Dataset class to load HDF5 data
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


# Testing functions
def test_data_loading():
    h5_file = "test_data.h5"
    create_test_h5_file(h5_file)

    dataset = SpeechDataset(h5_file)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    for mel_spectrograms, transcripts in dataloader:
        assert mel_spectrograms.shape[0] == 2
        assert len(transcripts) == 2
        print(f"Loaded mel spectrograms shape: {mel_spectrograms.shape}")
        print(f"Loaded transcripts shape: {transcripts.shape}")
        break

    print("Data loading test passed.")


def test_model_forward_pass():
    input_dim = 80
    hidden_dim = 128
    output_dim = len(characters)

    model = SpeechToTextModel(input_dim, hidden_dim, output_dim)
    mel_spectrogram = torch.rand(
        1, 80, 300
    )  # Batch size 1, 80 mel bands, 300 time steps

    output = model(mel_spectrogram)
    assert output.shape == (1, 300, output_dim)
    print(f"Model output shape: {output.shape}")

    print("Model forward pass test passed.")


def test_prediction():
    input_dim = 80
    hidden_dim = 128
    output_dim = len(characters)

    model = SpeechToTextModel(input_dim, hidden_dim, output_dim)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()

    audio_path = "sample_audio.pcm"

    # 샘플 오디오 파일 생성
    waveform = np.random.randn(16000).astype(np.float32)
    with open(audio_path, "wb") as f:
        f.write(waveform.tobytes())

    transcript = predict(model, audio_path, device="cpu")

    assert isinstance(transcript, str)
    print(f"Predicted transcript: {transcript}")
    print("Prediction test passed.")


if __name__ == "__main__":
    print("Running tests...")
    test_data_loading()
    test_model_forward_pass()
    test_prediction()
    print("All tests passed.")
