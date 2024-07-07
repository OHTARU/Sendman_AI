import os
import torch
import torchaudio
import numpy as np
from tqdm import tqdm

# 데이터 경로 설정
audio_folder = r'D:\한국어 음성\한국어_음성_분야\KsponSpeech_05\KsponSpeech_05'

# 모든 .pcm 파일 경로를 재귀적으로 찾기
audio_paths = []
transcript_paths = []
for root, dirs, files in os.walk(audio_folder):
    for file in files:
        if file.endswith('.pcm'):  # .pcm 파일 필터링
            audio_paths.append(os.path.join(root, file))
        elif file.endswith('.txt'):  # .txt 파일 필터링
            transcript_paths.append(os.path.join(root, file))

# .pcm 파일 로드 및 전처리
def load_pcm(file_path, channels=1, sample_rate=16000, dtype=np.int16):
    with open(file_path, 'rb') as f:
        pcm_data = np.frombuffer(f.read(), dtype=dtype)
        pcm_data = pcm_data.astype(np.float32) / np.iinfo(dtype).max
        if channels > 1:
            pcm_data = pcm_data.reshape(-1, channels)
    return pcm_data, sample_rate

# 특성 추출 (Mel-spectrogram)
def extract_features(waveform, sr, n_mels=80, n_fft=400):
    waveform = torch.tensor(waveform).unsqueeze(0)  # (1, N) 형태로 변환
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft)(waveform)
    return mel_spectrogram.squeeze(0)

# 텍스트 파일 로드
def load_transcript(file_path):
    encodings = ['utf-8', 'cp949', 'euc-kr']
    for enc in encodings:
        try:
            with open(file_path, 'r', encoding=enc) as f:
                transcript = f.read().strip()
            return transcript
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Failed to decode file {file_path} with available encodings")

# 파일을 처리하는 함수
def process_file(audio_path):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    transcript_path = next((path for path in transcript_paths if base_name in path), None)
    
    if transcript_path is None:
        return None
    
    # .pcm 파일 전처리
    waveform, sr = load_pcm(audio_path)
    mel_spectrogram = extract_features(waveform, sr)
    
    # 텍스트 파일 전처리
    transcript = load_transcript(transcript_path)
    
    return {
        'mel_spectrogram': mel_spectrogram,  # 이미 CPU로 이동
        'transcript': transcript
    }

# 데이터 처리 및 저장
def preprocess_data(audio_paths, transcript_paths):
    processed_data = []
    max_len = 0  # 최대 길이 초기화

    for audio_path in tqdm(audio_paths, desc="Processing audio files", unit="file"):
        result = process_file(audio_path)
        if result is not None:
            processed_data.append(result)
            if result['mel_spectrogram'].shape[1] > max_len:
                max_len = result['mel_spectrogram'].shape[1]

    return processed_data, max_len

if __name__ == '__main__':
    processed_data, max_len = preprocess_data(audio_paths, transcript_paths)
    # 여기서 데이터와 최대 길이를 저장할 수 있습니다.
    torch.save((processed_data, max_len), 'processed_data.pt')
    print("Processed data has been saved to 'processed_data.pt'.")
