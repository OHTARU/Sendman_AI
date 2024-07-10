import torch
from Let_modeling import TransformerModel, extract_features, load_pcm
import numpy as np

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

def load_pcm(file_path, channels=1, sample_rate=16000, dtype=np.int16):
    try:
        with open(file_path, 'rb') as f:
            pcm_data = f.read()
            data_size = len(pcm_data)
            element_size = np.dtype(dtype).itemsize
            print(f"File size: {data_size} bytes, Element size: {element_size} bytes")
            
            if data_size % element_size != 0:
                print(f"Warning: File size {data_size} is not a multiple of element size {element_size}. Padding with zeros.")
                # 패딩을 추가하여 크기를 맞춤
                pcm_data += b'\x00' * (element_size - data_size % element_size)
                data_size = len(pcm_data)
            
            pcm_data = np.frombuffer(pcm_data, dtype=dtype)
            if pcm_data.size % channels != 0:
                raise ValueError(f"PCM data size {pcm_data.size} is not a multiple of the number of channels {channels}")
            
            pcm_data = pcm_data.astype(np.float32) / np.iinfo(dtype).max
            if channels > 1:
                pcm_data = pcm_data.reshape(-1, channels)
        
        return pcm_data, sample_rate
    except Exception as e:
        print(f"Error loading PCM file: {e}")
        raise

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dim = 80
    hidden_dim = 512
    output_dim = len(characters)

    model = TransformerModel(input_dim, hidden_dim, output_dim).to(device)
    model.load_state_dict(torch.load('D:\\AI\\best_model.pth'))
    model.eval()

    audio_path = r"D:\AI\한국어 음성\평가용_데이터\eval_clean\KsponSpeech_E00001.pcm"
    waveform, sr = load_pcm(audio_path)
    print(f"Waveform shape: {waveform.shape}, Sample rate: {sr}")
    
    mel_spectrogram = extract_features(waveform, sr)
    print(f"Mel-spectrogram shape: {mel_spectrogram.shape}")

    mel_spectrogram = mel_spectrogram.unsqueeze(0).permute(2, 0, 1).to(device)
    with torch.no_grad():
        outputs = model(mel_spectrogram)
    
    print(f"Model outputs shape: {outputs.shape}")

    # Log probabilities 적용 후 argmax
    log_probs = outputs.log_softmax(2)
    outputs = log_probs.argmax(dim=2).squeeze(1)
    print(f"Outputs after argmax and squeeze: {outputs}")

    decoded_transcript = ''.join([index_to_char.get(idx.item(), '') for idx in outputs])
    print(f"Predicted transcript: {decoded_transcript}")

if __name__ == "__main__":
    main()
