import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import PatchDataset
from models_brevitas import QuantResSeg1D

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang chạy huấn luyện trên: {device}")

    # 1. Cấu hình Dữ liệu (Sửa lại tên file .h5 cho đúng với máy bạn)
    h5_file = "20211102T152000_patch1_highnoise_30_100_0_14.h5" 
    dataset = PatchDataset(h5_file, win_h=1, win_w=4000, stride_h=1, stride_w=2000)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

    # 2. Khởi tạo Mô hình Brevitas
    model = QuantResSeg1D(in_channels=1, num_classes=3).to(device)
    
    # 3. Định nghĩa Hàm Lỗi và Tối ưu hóa
    # Dùng BCEWithLogitsLoss vì mạng output ra số thực nguyên thủy, chưa qua Sigmoid
    criterion = nn.BCEWithLogitsLoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 4. Vòng lặp Huấn luyện (Train Loop)
    epochs = 10
    best_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for patches, masks in dataloader:
            patches, masks = patches.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(patches) # Đầu ra: (Batch, 3, Length)
            
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {avg_loss:.4f}")

        # Lưu lại mô hình tốt nhất
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), "deepdas_brevitas_best.pth")
            print("  -> Đã lưu mô hình tốt nhất!")

if __name__ == "__main__":
    main()