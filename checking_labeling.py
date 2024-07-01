import os
import json

audio_directory = r'D:\138.뉴스 대본 및 앵커 음성 데이터\01-1.정식개방데이터\Training\01.원천데이터\TS\SPK004'

def validate_and_process_audio_and_labels(folder_path):
    valid_data = []  # 유효한 데이터를 저장할 리스트
    invalid_data = []  # 유효하지 않은 데이터를 저장할 리스트

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
                        
                        # 필요한 필드가 모두 존재하는지 확인
                        required_fields = ['script', 'speaker', 'file_information']
                        if all(field in label_data for field in required_fields):
                            valid_data.append({
                                "audio_file_path": audio_file_path,
                                "script_id": label_data['script']['id'],
                                "script_text": label_data['script']['text'],
                                "speaker_id": label_data['speaker']['id'],
                                "speaker_age": label_data['speaker']['age'],
                                "speaker_sex": label_data['speaker']['sex'],
                                "audio_duration": label_data['file_information']['audio_duration'],
                                "audio_format": label_data['file_information']['audio_format'],
                            })
                        else:
                            invalid_data.append({
                                "audio_file_path": audio_file_path,
                                "label_file_path": label_file_path,
                                "reason": "Missing required fields"
                            })
                    except json.JSONDecodeError:
                        invalid_data.append({
                            "audio_file_path": audio_file_path,
                            "label_file_path": label_file_path,
                            "reason": "Invalid JSON format"
                        })
                else:
                    invalid_data.append({
                        "audio_file_path": audio_file_path,
                        "label_file_path": label_file_path,
                        "reason": "Label file does not exist"
                    })
    
    return valid_data, invalid_data

# 데이터 처리 및 검증
valid_data, invalid_data = validate_and_process_audio_and_labels(audio_directory)

# 유효한 데이터 출력
print("Valid Data:")
for entry in valid_data:
    print(f"Audio File: {entry['audio_file_path']}, Script Id: {entry['script_id']}, Script Text: {entry['script_text']},")
    print(f"Speaker ID: {entry['speaker_id']}, Speaker Age: {entry['speaker_age']}, Speaker Sex: {entry['speaker_sex']},")
    print(f"Audio Format: {entry['audio_format']}, Audio Duration: {entry['audio_duration']} seconds")

# 유효하지 않은 데이터 출력
print("\nInvalid Data:")
for entry in invalid_data:
    print(f"Audio File: {entry['audio_file_path']}, Label File: {entry['label_file_path']}, Reason: {entry['reason']}")
