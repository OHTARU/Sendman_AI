import h5py
import torch
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


# 멜 스펙트로그램과 전사 데이터를 HDF5 파일에 저장하는 함수
def process_and_save(idx, item, mel_spectrograms, transcripts, max_len):
    mel = item["mel_spectrogram"].numpy()
    mel_spectrograms[idx, :, : mel.shape[1]] = mel
    transcripts[idx] = item["transcript"]
    return idx


# 데이터를 HDF5 파일로 저장하는 함수
def save_to_h5(data, h5_file, max_len):
    with h5py.File(h5_file, "w") as f:
        # 멜 스펙트로그램 데이터셋 생성
        mel_spectrograms = f.create_dataset(
            "mel_spectrograms",
            (len(data), 80, max_len),
            dtype="float32",
            chunks=(1, 80, max_len),  # 청크 크기 설정
            compression="gzip",  # 압축 사용
        )
        # 전사 데이터셋 생성
        transcripts = f.create_dataset(
            "transcripts",
            (len(data),),
            dtype=h5py.special_dtype(vlen=str),
            compression="gzip",  # 압축 사용
        )

        # 스레드를 사용하여 데이터를 병렬로 저장
        with ThreadPoolExecutor(max_workers=4) as executor:  # 최대 4개의 스레드 사용
            futures = [
                executor.submit(
                    process_and_save, idx, item, mel_spectrograms, transcripts, max_len
                )
                for idx, item in enumerate(data)
            ]
            for future in tqdm(
                as_completed(futures),
                total=len(data),
                desc="Saving to HDF5",
                unit="sample",
            ):
                future.result()


if __name__ == "__main__":
    # 전처리된 데이터 로드
    processed_data, max_len = torch.load("D:\\AI\\processed_data.pt")

    # 데이터 확인 (디버깅 및 검증)
    if not isinstance(processed_data, list):
        raise TypeError(
            f"Expected processed_data to be a list, got {type(processed_data).__name__}."
        )
    if not all(
        isinstance(item, dict) and "mel_spectrogram" in item and "transcript" in item
        for item in processed_data
    ):
        raise ValueError(
            "Each item in processed_data should be a dict with 'mel_spectrogram' and 'transcript' keys."
        )

    # HDF5 파일로 저장
    save_to_h5(processed_data, "D:\\AI\\processed_data.h5", max_len)
    print("Processed data has been saved to HDF5.")
