import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model

# 모델 요약 출력
print("Model Summary:")
model.summary()

# 모델 저장 및 로드
model_path = 'stt_model.keras'
model.save(model_path)
loaded_model = load_model(model_path)
print("Model loaded successfully!")

# 학습 곡선 시각화 함수
def plot_history(history):
    plt.figure(figsize=(12, 4))
    
    # 학습 및 검증 손실
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    # 학습 및 검증 정확도
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='train_accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.show()

# 학습 기록 그래프 출력
plot_history(history)

# 예측 수행
predictions = loaded_model.predict(X_test)

# 예측 결과 확인
for i, prediction in enumerate(predictions[:5]):
    predicted_label = np.argmax(prediction)
    actual_label = np.argmax(y_test[i])
    print(f"Predicted: {predicted_label}, Actual: {actual_label}")
