from flask import Flask, request, jsonify
import easyocr
from werkzeug.utils import secure_filename
import concurrent.futures
import os

app = Flask(__name__)
reader = easyocr.Reader(["en", "ko"])

# 허용할 파일 확장자 목록
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
# 최대 파일 크기 설정 (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
# 타임아웃 시간 설정 (30초)
TIMEOUT = 30  # 30초


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# 텍스트 추출 함수
def process_image(image_data):
    result = reader.readtext(image_data)
    texts = [text[1] for text in result]
    return texts


@app.route("/ocr", methods=["POST"])
def ocr():
    try:
        if "image" not in request.files:
            return jsonify({"error": "이미지 파일이 제공되지 않았습니다."}), 400

        image_file = request.files["image"]

        # 파일 크기 검증
        image_file.seek(0, os.SEEK_END)  # 파일 포인터를 파일 끝으로 이동
        file_size = image_file.tell()
        image_file.seek(0)  # 파일 포인터를 다시 처음으로 이동

        if file_size > MAX_FILE_SIZE:
            return jsonify({"error": "파일 크기가 10MB를 초과합니다."}), 400

        # 파일 형식 검증
        if not allowed_file(image_file.filename):
            return (
                jsonify(
                    {
                        "error": "허용되지 않는 파일 형식입니다. PNG, JPG 또는 JPEG 이미지를 업로드하세요."
                    }
                ),
                400,
            )

        try:
            # 이미지 처리 비동기 실행
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(process_image, image_file.read())
                texts = future.result(timeout=TIMEOUT)

            return jsonify({"texts": texts}), 200
        except concurrent.futures.TimeoutError:
            return (
                jsonify({"error": "이미지 처리 시간이 30초를 초과했습니다."}),
                504,
            )
        except Exception as e:
            return (
                jsonify({"error": f"이미지 처리 중 오류가 발생했습니다: {str(e)}"}),
                500,
            )

    except Exception as e:
        return jsonify({"error": f"예상치 못한 오류가 발생했습니다: {str(e)}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
