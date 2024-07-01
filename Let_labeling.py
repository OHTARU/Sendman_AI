import os
import json
from tqdm import tqdm

audio_directory = r'E:\138.뉴스 대본 및 앵커 음성 데이터\01-1.정식개방데이터\Training\01.원천데이터\TS'

def process_audio_and_labels(folder_path):
    data = []  # 수집한 데이터를 저장할 리스트
    total_files = sum([len(files) for r, d, files in os.walk(folder_path)])
    with tqdm(total=total_files, desc="Processing files", unit="file") as pbar:
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_file_path = os.path.join(root, file)
                    label_file_name = file.replace('.wav', '.json')
                    label_file_path = os.path.join(root, label_file_name)
                    
                    if os.path.exists(label_file_path):
                        try:
                            with open(label_file_path, 'r', encoding='utf-8') as f:
                                label_data = json.load(f)

                            data.append({
                                "audio_file_path": audio_file_path,
                                "script_id": label_data['script']['id'],
                                "script_text": label_data['script']['text'],
                                "speaker_id": label_data['speaker']['id'],
                                "speaker_age": label_data['speaker']['age'],
                                "speaker_sex": label_data['speaker']['sex'],
                                "audio_duration": label_data['file_information']['audio_duration'],
                                "audio_format": label_data['file_information']['audio_format'],
                            })
                        except PermissionError:
                            print(f"Permission denied: {label_file_path}")
                        except Exception as e:
                            print(f"Error reading {label_file_path}: {e}")
                    else:
                        print(f"Label file does not exist: {label_file_name}")
                    
                    pbar.update(1)  # Update the progress bar
    
    return data

# 데이터 처리 및 출력
processed_data = process_audio_and_labels(audio_directory)
for entry in processed_data:
    print(f"Audio File: {entry['audio_file_path']}, Script Id: {entry['script_id']}, Script Text: {entry['script_text']},")
    print(f"Speaker ID: {entry['speaker_id']}, Speaker Age: {entry['speaker_age']}, Speaker Sex: {entry['speaker_sex']},")
    print(f"Audio Format: {entry['audio_format']}, Audio Duration: {entry['audio_duration']} seconds")
