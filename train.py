import scipy.io
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import os

# --- CẤU HÌNH ---
DATA_FILE = 'MMF_trainData.mat' 
INPUT_DIM = 100  # 10x10 pixels

def load_data():
    if not os.path.exists(DATA_FILE):
        print(f"LỖI: Không tìm thấy file '{DATA_FILE}'")
        return None, None, None

    print(f"--> Đang đọc dữ liệu từ {DATA_FILE}...")
    data = scipy.io.loadmat(DATA_FILE)

    # 1. Xử lý dữ liệu đầu vào (X)
    X_raw = np.transpose(data['X_data'], (2, 0, 1))
    
    # 2. Xử lý nhãn (Y)
    Y_rho = data['Y_label_Rho']
    Y_theta = data['Y_label_Theta']
    
    if Y_rho.shape[0] == 1:
        Y_rho = Y_rho.T
    if Y_theta.shape[0] == 1:
        Y_theta = Y_theta.T

    Y = np.concatenate([Y_rho, Y_theta], axis=1)

    print(f"    Dữ liệu gốc (X): {X_raw.shape}")
    print(f"    Dữ liệu nhãn (Y): {Y.shape}")

    # 3. Resize ảnh về 10x10
    print("--> Đang nén ảnh xuống 10x10...")
    X_raw = X_raw[..., np.newaxis]  # (N, 30, 30, 1)
    X_resized = tf.image.resize(X_raw, [10, 10]).numpy()
    
    # Chuẩn hóa X về [0, 1]
    X_norm = X_resized / np.max(X_resized)
    
    # Duỗi phẳng: (N, 10, 10, 1) -> (N, 100, 1) cho 1D-CNN
    X_flat = X_norm.reshape(X_norm.shape[0], INPUT_DIM, 1)
    
    print(f"    Input shape sau khi xử lý: {X_flat.shape}")
    
    # 4. Chuẩn hóa Y (Rho/Theta) về [0,1]
    print("--> Đang chuẩn hóa nhãn Y...")
    scaler_Y = MinMaxScaler()
    Y_norm = scaler_Y.fit_transform(Y)
    
    return X_flat, Y_norm, scaler_Y

def build_model_100xn(output_size):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(INPUT_DIM, 1)))  # (100, 1) cho 1D-CNN
    
    # Conv1D Layer 1
    model.add(layers.Conv1D(32, kernel_size=5, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Conv1D Layer 2
    model.add(layers.Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(pool_size=2))
    
    # Flatten và Dense
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))  # Lớp 100 neuron như "100xn"
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(output_size, activation='linear'))
    
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # 1. Load dữ liệu
    X, Y_norm, scaler_Y = load_data()
    
    if X is not None:
        # 2. Chia dữ liệu Train (80%) và Test (20%)
        print("\n--> Đang chia dữ liệu Train/Test...")
        X_train, X_test, y_train, y_test = train_test_split(X, Y_norm, test_size=0.2, random_state=42)
        
        # 3. Khởi tạo mô hình
        print(f"--> Khởi tạo mô hình với {Y_norm.shape[1]} đầu ra...")
        model = build_model_100xn(output_size=Y_norm.shape[1])
        model.summary()
        
        # 4. Train mô hình với early stopping
        print("\n--> BẮT ĐẦU TRAINING...")
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        history = model.fit(X_train, y_train, 
                            epochs=100,  # Tăng để tốt hơn
                            batch_size=32, 
                            validation_split=0.1, 
                            callbacks=[early_stop],
                            verbose=1)
        
        # 5. Đánh giá trên Test Set
        print("\n--> ĐÁNH GIÁ TRÊN TEST SET...")
        test_loss, test_mae = model.evaluate(X_test, y_test)
        print(f"    Sai số Test (MSE): {test_loss:.4f}")
        print(f"    Sai số tuyệt đối trung bình (MAE): {test_mae:.4f}")
        
        # Dự đoán và inverse scale về giá trị gốc
        predictions_norm = model.predict(X_test)
        predictions = scaler_Y.inverse_transform(predictions_norm)  # Inverse về Rho/Theta gốc
        y_test_orig = scaler_Y.inverse_transform(y_test)
        
        print("\nDự đoán vs Thực tế (5 mẫu đầu - giá trị gốc):")
        for i in range(5):
            print(f"Mẫu {i+1}: Dự đoán [Rho, Theta] = {predictions[i]}, Thực tế = {y_test_orig[i]}")
        
        # 6. Vẽ biểu đồ loss
        plt.figure(figsize=(10, 5))
        plt.plot(history.history['loss'], label='Sai số trên tập Train')
        plt.plot(history.history['val_loss'], label='Sai số trên tập Validation')
        plt.title('Biểu đồ huấn luyện (Loss)')
        plt.xlabel('Số lần học (Epoch)')
        plt.ylabel('Sai số (MSE)')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        # 7. Vẽ scatter plot dự đoán vs thực tế (cho Rho và Theta)
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.scatter(y_test_orig[:, 0], predictions[:, 0])
        plt.plot([min(y_test_orig[:, 0]), max(y_test_orig[:, 0])], [min(y_test_orig[:, 0]), max(y_test_orig[:, 0])], 'r--')
        plt.title('Dự đoán Rho vs Thực tế')
        plt.xlabel('Thực tế')
        plt.ylabel('Dự đoán')
        
        plt.subplot(1, 2, 2)
        plt.scatter(y_test_orig[:, 1], predictions[:, 1])
        plt.plot([min(y_test_orig[:, 1]), max(y_test_orig[:, 1])], [min(y_test_orig[:, 1]), max(y_test_orig[:, 1])], 'r--')
        plt.title('Dự đoán Theta vs Thực tế')
        plt.xlabel('Thực tế')
        plt.ylabel('Dự đoán')
        plt.show()
        
        # 8. Lưu model
        model.save('mmf_model.h5')
        print("Model đã lưu thành 'mmf_model.h5'")
        
        print("\nTôi đã ở đây :D")
