import torch
from Let_modeling import TransformerModel, extract_features, load_pcm
import os

# 한글을 포함한 허용 문자 정의
hangul_chars = [chr(i) for i in range(ord('가'), ord('힣') + 1)]
allowed_characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+[{]}\\|;:'\",<.>/?`~ " + ''.join(hangul_chars)

# allowed_characters 길이를 11267로 설정
if len(allowed_characters) < 11267:
    for i in range(11267 - len(allowed_characters)):
        allowed_characters += chr(1000 + i)
elif len(allowed_characters) > 11267:
    allowed_characters = allowed_characters[:11267]

characters = allowed_characters
char_to_index = {char: idx for idx, char in enumerate(characters)}
index_to_char = {idx: char for idx, char in enumerate(characters)}

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = 80
    hidden_dim = 512
    output_dim = len(characters)

    model = TransformerModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load('D:\\AI\\best_model.pth'))
    model.eval()

    audio_path = r"D:\한국어 음성\한국어_음성_분야\KsponSpeech_03\KsponSpeech_03\KsponSpeech_0249\KsponSpeech_248029.pcm"
    waveform, sr = load_pcm(audio_path)
    print(f"Waveform shape: {waveform.shape}, Sample rate: {sr}")
    
    mel_spectrogram = extract_features(waveform, sr)
    print(f"Mel-spectrogram shape: {mel_spectrogram.shape}")

    mel_spectrogram = mel_spectrogram.unsqueeze(0).permute(2, 0, 1).to(device)
    with torch.no_grad():
        outputs = model(mel_spectrogram)
    
    print(f"Model outputs shape: {outputs.shape}")

    outputs = outputs.argmax(dim=2).squeeze(1)
    print(f"Outputs after argmax and squeeze: {outputs}")

    decoded_transcript = ''.join([index_to_char.get(idx, '') for idx in outputs])
    print(f"Predicted transcript: {decoded_transcript}")

if __name__ == "__main__":
    main()
