# test.py
import os
import numpy as np
import torch
from tqdm import tqdm
import torchaudio


# 더미 데이터 생성 코드
def generate_dummy_data(base_dir="test_data", num_files=5):
    audio_dir = os.path.join(base_dir, "audio")
    transcript_dir = os.path.join(base_dir, "transcripts")

    # 디렉토리 생성
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(transcript_dir, exist_ok=True)

    # 더미 PCM 파일과 TXT 파일 생성
    for i in range(num_files):
        # 더미 PCM 파일 생성
        audio_path = os.path.join(audio_dir, f"file_{i}.pcm")
        sample_rate = 16000
        duration = 2  # 2초 길이
        num_samples = sample_rate * duration
        waveform = np.random.uniform(-1, 1, num_samples).astype(
            np.float32
        )  # 무작위 신호
        waveform.tofile(audio_path)  # PCM 파일로 저장

        # 더미 TXT 파일 생성
        transcript_path = os.path.join(transcript_dir, f"file_{i}.txt")
        transcript = f"Sample text {i} 한글 문장"  # 더미 텍스트
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript)


# 데이터 전처리 함수들
def load_pcm(file_path, channels=1, sample_rate=16000, dtype=np.int16):
    with open(file_path, "rb") as f:
        pcm_data = np.frombuffer(f.read(), dtype=dtype)
        pcm_data = pcm_data.astype(np.float32) / np.iinfo(dtype).max
        if channels > 1:
            pcm_data = pcm_data.reshape(-1, channels)
    return pcm_data, sample_rate


def extract_features(waveform, sr, n_mels=80, n_fft=400):
    waveform = torch.tensor(waveform).unsqueeze(0)  # (1, N) 형태로 변환
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft
    )(waveform)
    return mel_spectrogram.squeeze(0)


def load_transcript(file_path):
    encodings = ["utf-8", "cp949", "euc-kr"]
    for enc in encodings:
        try:
            with open(file_path, "r", encoding=enc) as f:
                transcript = f.read().strip()
            return transcript
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Failed to decode file {file_path} with available encodings")


def text_to_indices(text):
    characters = allowed_characters
    char_to_index = {char: idx for idx, char in enumerate(characters)}
    return [char_to_index[char] for char in text if char in char_to_index]


def process_file(audio_path, transcript_paths):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    transcript_path = next(
        (path for path in transcript_paths if base_name in path), None
    )

    if transcript_path is None:
        return None

    # .pcm 파일 전처리
    waveform, sr = load_pcm(audio_path)
    mel_spectrogram = extract_features(waveform, sr)

    # 텍스트 파일 전처리
    transcript = load_transcript(transcript_path)
    transcript_indices = text_to_indices(transcript)

    return {
        "mel_spectrogram": mel_spectrogram,  # 이미 CPU로 이동
        "transcript": transcript,
        "transcript_indices": transcript_indices,
    }


def preprocess_data(audio_paths, transcript_paths):
    processed_data = []
    max_len = 0  # 최대 길이 초기화

    for audio_path in tqdm(audio_paths, desc="Processing audio files", unit="file"):
        result = process_file(audio_path, transcript_paths)
        if result is not None:
            processed_data.append(result)
            if result["mel_spectrogram"].shape[1] > max_len:
                max_len = result["mel_spectrogram"].shape[1]

    return processed_data, max_len


# 문자 집합 정의
hangul_chars = [chr(i) for i in range(ord("가"), ord("힣") + 1)]
allowed_characters = (
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    "!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?~ " + "".join(hangul_chars)
)
allowed_characters += " "

if len(allowed_characters) < 11267:
    for i in range(11267 - len(allowed_characters)):
        allowed_characters += chr(1000 + i)
elif len(allowed_characters) > 11267:
    allowed_characters = allowed_characters[:11267]

# 1. 더미 데이터 생성
generate_dummy_data(base_dir="test_data", num_files=5)

# 2. 데이터 경로 설정
audio_folder = os.path.join("test_data", "audio")
transcript_folder = os.path.join("test_data", "transcripts")

# 모든 .pcm 파일 경로를 재귀적으로 찾기
audio_paths = []
transcript_paths = []
for root, dirs, files in os.walk(audio_folder):
    for file in files:
        if file.endswith(".pcm"):  # .pcm 파일 필터링
            audio_paths.append(os.path.join(root, file))
for root, dirs, files in os.walk(transcript_folder):
    for file in files:
        if file.endswith(".txt"):  # .txt 파일 필터링
            transcript_paths.append(os.path.join(root, file))

# 3. 데이터 처리 및 저장
processed_data, max_len = preprocess_data(audio_paths, transcript_paths)
# 데이터와 최대 길이를 저장
torch.save((processed_data, max_len), "test_data/processed_data.pt")
print("Processed data has been saved to 'test_data/processed_data.pt'.")
