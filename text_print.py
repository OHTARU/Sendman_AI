import argparse
import torch
import torch.nn as nn
import numpy as np
import torchaudio
from torch import Tensor
from kospeech.vocabs.ksponspeech import KsponSpeechVocabulary
from kospeech.models.deepspeech2.model import DeepSpeech2

def parse_audio(audio_path: str) -> Tensor:
    # 파일 확장자 확인
    file_extension = audio_path.split('.')[-1].lower()
    
    # .acc 파일도 지원
    if file_extension not in ['wav', 'acc']:
        raise ValueError("Unsupported audio format. Please use a .wav or .acc file.")
    
    # 오디오 파일 불러오기
    waveform, sample_rate = torchaudio.load(audio_path)
    
    # 샘플링 속도 확인 및 리샘플링
    if sample_rate != 16000:
        resample_transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample_transform(waveform)
    
    # FBANK 특징 추출
    feature = torchaudio.compliance.kaldi.fbank(
        waveform=waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        window_type='hamming',
        sample_frequency=16000
    ).transpose(0, 1).numpy()

    # 정규화
    feature -= feature.mean()
    feature /= np.std(feature)

    return torch.FloatTensor(feature).transpose(0, 1)

def main(model_path: str, audio_path: str):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 오디오 전처리
    feature = parse_audio(audio_path)
    input_length = torch.LongTensor([len(feature)])
    
    # 단어장 로드
    vocab = KsponSpeechVocabulary(r"D:\AI\한국어_음성\한국어_음성_분야\aihub_labels.csv")

    # 모델 로드
    model = torch.load(model_path, map_location=device).to(device)
    if isinstance(model, nn.DataParallel):
        model = model.module
    model.eval()

    # 모델 추론
    with torch.no_grad():
        inputs_tensor = feature.unsqueeze(0).to(device)
        input_lengths_tensor = input_length.to(device)
        y_hats, _ = model(inputs_tensor, input_lengths_tensor)

    # 결과 디코딩
    sentence = vocab.label_to_string(y_hats[0].argmax(dim=-1).cpu().detach().numpy())
    print(sentence)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KoSpeech DeepSpeech2 Inference')
    parser.add_argument('--model_path', type=str, required=True, help="Path to the trained model file")
    parser.add_argument('--audio_path', type=str, required=True, help="Path to the input audio file")
    args = parser.parse_args()
    
    main(args.model_path, args.audio_path)
