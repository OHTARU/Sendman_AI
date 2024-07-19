# import os

# def count_pcm_files(directory):
#     """
#     Counts the number of .pcm files in the specified directory.

#     Args:
#         directory (str): The path to the directory to search.

#     Returns:
#         int: The number of .pcm files in the directory.
#     """
#     pcm_count = 0
#     for filename in os.listdir(directory):
#         if filename.endswith('.pcm'):
#             pcm_count += 1
#     return pcm_count

# # Example usage:
# directory_path = r"D:\AI\한국어_음성\한국어_음성_분야\KsponSpeech_01"
# pcm_files_count = count_pcm_files(directory_path)
# print(f'The number of .pcm files in the directory is: {pcm_files_count}')
# import os

# def count_pcm_files_in_subfolders(directory):
#     """
#     Counts the number of .pcm files in the specified directory and its subdirectories.

#     Args:
#         directory (str): The path to the directory to search.

#     Returns:
#         int: The number of .pcm files in the directory and its subdirectories.
#     """
#     pcm_count = 0
#     for root, dirs, files in os.walk(directory):
#         for filename in files:
#             if filename.endswith('.pcm'):
#                 pcm_count += 1
#     return pcm_count

# # Example usage:
# directory_path = r"D:\AI\한국어_음성\한국어_음성_분야"
# pcm_files_count = count_pcm_files_in_subfolders(directory_path)
# print(f'The number of .pcm files in the directory and its subdirectories is: {pcm_files_count}')

# total_files = 622545
# train_num = int(total_files * 0.85)  # 80% for training
# valid_num = total_files - train_num  # Remaining 20% for validation

# print(f'Training set size: {train_num}')
# print(f'Validation set size: {valid_num}')

# import torch

# # 모델 파일 로드
# model_path = r'C:\Users\bak10\Desktop\kospeech-latest\outputs\2024-07-15\18-01-11\model.pt'
# model_state_dict = torch.load(model_path, map_location=torch.device('cpu'))

# # DataParallel 객체 해제
# if isinstance(model_state_dict, torch.nn.DataParallel):
#     model_state_dict = model_state_dict.module.state_dict()

# # 모델의 키와 각 파라미터의 크기 출력
# for param_tensor in model_state_dict:
#     print(param_tensor, "\t", model_state_dict[param_tensor].size())


# import wave
# import numpy as np

# def pcm_to_wav(pcm_file, wav_file, channels=1, bit_depth=16, sample_rate=16000):
#     # PCM 파일 읽기
#     with open(pcm_file, 'rb') as pcmfile:
#         pcm_data = pcmfile.read()

#     # PCM 데이터를 numpy 배열로 변환
#     pcm_array = np.frombuffer(pcm_data, dtype=np.int16)

#     # WAV 파일 생성
#     with wave.open(wav_file, 'wb') as wavfile:
#         wavfile.setnchannels(channels)
#         wavfile.setsampwidth(bit_depth // 8)
#         wavfile.setframerate(sample_rate)
#         wavfile.writeframes(pcm_array.tobytes())

# # 사용 예제
# pcm_file_path = r"D:\AI\한국어_음성\한국어_음성_분야\KsponSpeech_01\KsponSpeech_0001\KsponSpeech_000001.pcm"
# wav_file_path = r'C:\Users\bak10\Desktop\kospeech-latest\file.wav'
# pcm_to_wav(pcm_file_path, wav_file_path)
import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch import Tensor

from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.data.audio.core import load_audio
from kospeech.models import (
    SpeechTransformer,
    Jasper,
    DeepSpeech2,
    ListenAttendSpell,
    Conformer,
)

def parse_audio(audio_path: str, del_silence: bool = False, audio_extension: str = 'pcm') -> Tensor:
    signal = load_audio(audio_path, del_silence, extension=audio_extension)
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=Tensor(signal).unsqueeze(0),
        num_mel_bins=80,
        frame_length=25,  # Adjusted to 25ms frame length
        frame_shift=10,
        window_type='hamming'
    ).transpose(0, 1).numpy()

    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)

parser = argparse.ArgumentParser(description='KoSpeech')
parser.add_argument('--model_path', type=str, required=True)
parser.add_argument('--audio_path', type=str, required=True)
opt = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

feature = parse_audio(opt.audio_path, del_silence=True)
input_length = torch.LongTensor([len(feature)])
vocab = KsponSpeechVocabulary(r"D:\AI\한국어_음성\한국어_음성_분야\aihub_labels.csv")

model = torch.load(opt.model_path, map_location=lambda storage, loc: storage).to(device)
if isinstance(model, nn.DataParallel):
    model = model.module
model.eval()

if isinstance(model, ListenAttendSpell):
    model.encoder.device = device
    model.decoder.device = device

    y_hats = model.recognize(feature.unsqueeze(0).to(device), input_length.to(device))
elif isinstance(model, DeepSpeech2):
    model.device = device
    y_hats = model.recognize(feature.unsqueeze(0).to(device), input_length.to(device))
elif isinstance(model, SpeechTransformer) or isinstance(model, Jasper) or isinstance(model, Conformer):
    y_hats = model.recognize(feature.unsqueeze(0).to(device), input_length.to(device))

sentence = vocab.label_to_string(y_hats.cpu().detach().numpy())
print(sentence)


# import torch

# model_path = r"C:\Users\bak10\Desktop\kospeech-latest\outputs\2024-07-15\23-36-13\model.pt"
# state_dict = torch.load(model_path, map_location='cpu')

# # DataParallel로 저장된 모델을 로드할 때
# if isinstance(state_dict, torch.nn.DataParallel):
#     state_dict = state_dict.module.state_dict()
# elif isinstance(state_dict, dict) and 'state_dict' in state_dict:
#     state_dict = state_dict['state_dict']

# # 모델의 레이어 이름과 형태를 출력합니다.
# for key, value in state_dict.items():
#     print(f"{key}: {value.shape}")
