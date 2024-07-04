import os
import torch
import torchaudio
import numpy as np
import pandas as pd
from tqdm import tqdm  # tqdm 라이브러리 추가
import pickle

# GPU 사용 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 데이터 경로 설정
audio_folder = r'D:\한국어 음성\한국어_음성_분야\KsponSpeech_05\KsponSpeech_05\KsponSpeech_0497'
transcript_folder = r'D:\한국어 음성\한국어_음성_분야\KsponSpeech_05\KsponSpeech_05\KsponSpeech_0497'

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
    waveform = torch.tensor(waveform).unsqueeze(0).to(device)  # (1, N) 형태로 변환하고 GPU로 전송
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_mels=n_mels, n_fft=n_fft).to(device)(waveform)
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

# 모든 데이터를 전처리하여 저장
processed_data = []
for audio_path in tqdm(audio_paths, desc="Processing audio files"):
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    transcript_path = next((path for path in transcript_paths if base_name in path), None)
    
    if transcript_path is None:
        continue
    
    # .pcm 파일 전처리
    waveform, sr = load_pcm(audio_path)
    mel_spectrogram = extract_features(waveform, sr)
    
    # 텍스트 파일 전처리
    transcript = load_transcript(transcript_path)
    
    # 전처리된 데이터 저장
    processed_data.append({
        'mel_spectrogram': mel_spectrogram.cpu(),  # GPU에서 CPU로 이동
        'transcript': transcript
    })

# 전처리된 데이터 확인
print(processed_data[0]['mel_spectrogram'].shape)
print(processed_data[0]['transcript'])

# 처리된 데이터를 파일로 저장
with open('processed_data.pkl', 'wb') as f:
    pickle.dump(processed_data, f)

print("Processed data has been saved.")


# import os
# import torch
# import torchaudio
# import numpy as np
# import pandas as pd
# from tqdm import tqdm
# from multiprocessing import Pool, cpu_count
# import pickle

# # 데이터 경로 설정
# audio_folder = r'D:\한국어 음성\한국어_음성_분야\KsponSpeech_05\KsponSpeech_05\KsponSpeech_0497'
# transcript_folder = r'D:\한국어 음성\한국어_음성_분야\KsponSpeech_05\KsponSpeech_05\KsponSpeech_0497'

# # 모든 .pcm 파일 경로를 재귀적으로 찾기
# audio_paths = []
# transcript_paths = []
# for root, dirs, files in os.walk(audio_folder):
#     for file in files:
#         if file.endswith('.pcm'):  # .pcm 파일 필터링
#             audio_paths.append(os.path.join(root, file))
#         elif file.endswith('.txt'):  # .txt 파일 필터링
#             transcript_paths.append(os.path.join(root, file))

# # .pcm 파일 로드 및 전처리
# def load_pcm(file_path, channels=1, sample_rate=16000, dtype=np.int16):
#     with open(file_path, 'rb') as f:
#         pcm_data = np.frombuffer(f.read(), dtype=dtype)
#         pcm_data = pcm_data.astype(np.float32) / np.iinfo(dtype).max
#         if channels > 1:
#             pcm_data = pcm_data.reshape(-1, channels)
#     return pcm_data, sample_rate

# # 특성 추출 (Mel-spectrogram)
# def extract_features(waveform, sr, n_mels=80, n_fft=400):
#     waveform = torch.tensor(waveform).unsqueeze(0)  # (1, N) 형태로 변환
#     mel_spectrogram = torchaudio.transforms.MelSpectrogram(
#         sample_rate=sr, n_mels=n_mels, n_fft=n_fft)(waveform)
#     return mel_spectrogram.squeeze(0)

# # 텍스트 파일 로드
# def load_transcript(file_path):
#     encodings = ['utf-8', 'cp949', 'euc-kr']
#     for enc in encodings:
#         try:
#             with open(file_path, 'r', encoding=enc) as f:
#                 transcript = f.read().strip()
#             return transcript
#         except UnicodeDecodeError:
#             continue
#     raise ValueError(f"Failed to decode file {file_path} with available encodings")

# # 파일을 처리하는 함수
# def process_file(audio_path):
#     base_name = os.path.splitext(os.path.basename(audio_path))[0]
#     transcript_path = next((path for path in transcript_paths if base_name in path), None)
    
#     if transcript_path is None:
#         return None
    
#     # .pcm 파일 전처리
#     waveform, sr = load_pcm(audio_path)
#     mel_spectrogram = extract_features(waveform, sr)
    
#     # 텍스트 파일 전처리
#     transcript = load_transcript(transcript_path)
    
#     return {
#         'mel_spectrogram': mel_spectrogram,  # 이미 CPU로 이동
#         'transcript': transcript
#     }

# # 멀티프로세싱을 사용하여 데이터 처리
# if __name__ == '__main__':
#     with Pool(cpu_count()) as p:
#         results = list(tqdm(p.imap(process_file, audio_paths), total=len(audio_paths)))

#     # 결과에서 None 값을 제거
#     processed_data = [result for result in results if result is not None]

#     # 전처리된 데이터 확인
#     print(processed_data[0]['mel_spectrogram'].shape)
#     print(processed_data[0]['transcript'])

#     # 처리된 데이터를 파일로 저장
#     with open('processed_data.pkl', 'wb') as f:
#         pickle.dump(processed_data, f)

#     print("Processed data has been saved.")
