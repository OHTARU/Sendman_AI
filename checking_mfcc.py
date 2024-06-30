import numpy as np
import os

def check_preprocessed_data(output_folder):
    total_files = 0
    total_size = 0
    for root, _, files in os.walk(output_folder):
        for file in files:  
            if file.endswith('_features.npy'):
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                total_files += 1
                total_size += file_size
                print(f"File: {file}, Size: {file_size / (1024 * 1024):.2f} MB")
    
    print(f"\nTotal Files: {total_files}")
    print(f"Total Size: {total_size / (1024 * 1024):.2f} MB")

output_directory = r'D:\138.뉴스 대본 및 앵커 음성 데이터\새 폴더'
check_preprocessed_data(output_directory)
