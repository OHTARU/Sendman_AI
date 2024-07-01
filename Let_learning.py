import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Masking, Input, GlobalAveragePooling1D, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

def load_preprocessed_data(data_folder, max_len=100):
    features = []
    labels = []
    for root, _, files in os.walk(data_folder):
        for file in files:
            if file.endswith('_features.npy'):
                feature_path = os.path.join(root, file)
                feature_data = np.load(feature_path, allow_pickle=True)
                
                if feature_data.shape[2] != max_len:
                    continue

                if feature_data.shape[1] != 13:
                    continue
                
                features.extend(feature_data)
                
                if 'KBS' in file:
                    label = 0
                elif 'YTN' in file:
                    label = 1
                elif 'MBC' in file:
                    label = 2
                elif 'OBS' in file:
                    label = 3
                elif 'SBS' in file:
                    label = 4
                else:
                    raise ValueError(f"Unknown label for file: {file}")
                
                labels.extend([label] * len(feature_data))
    
    return features, labels

# Data preparation
data_folder = r'D:\138.뉴스 대본 및 앵커 음성 데이터\MFCC'
X, y = load_preprocessed_data(data_folder, max_len=100)

# 데이터가 올바르게 로드되었는지 확인
if len(X) == 0 or len(y) == 0:
    raise ValueError("No data loaded. Check the data folder and preprocessing steps.")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

# Normalize the data
if X.size != 0:
    X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
else:
    raise ValueError("Loaded data is empty. Cannot normalize empty data.")

# One-hot encode labels
y = tf.keras.utils.to_categorical(y, num_classes=5)

# Split into train and test datasets
if X.shape[0] > 0 and y.shape[0] > 0:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
else:
    raise ValueError("No data available for train-test split.")

# Build the model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(Masking(mask_value=0.0))
model.add(LSTM(128, return_sequences=True))
model.add(BatchNormalization())
model.add(GlobalAveragePooling1D())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(5, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=30, batch_size=64, validation_data=(X_test, y_test))

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Save the model (Keras format)
model_path = 'stt_model.keras'
model.save(model_path)

# Load and evaluate the saved model
loaded_model = load_model(model_path)
loss, accuracy = loaded_model.evaluate(X_test, y_test)
print(f'Loaded model Test Loss: {loss}, Test Accuracy: {accuracy}')

# Predict on test data
predictions = loaded_model.predict(X_test)

# Print predictions and actual labels for first 5 samples
for i, prediction in enumerate(predictions[:5]):
    predicted_label = np.argmax(prediction)
    actual_label = np.argmax(y_test[i])
    print(f"Predicted: {predicted_label}, Actual: {actual_label}")
