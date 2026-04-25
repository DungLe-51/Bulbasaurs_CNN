import torch
import brevitas.onnx as bo
from models_brevitas import QuantResSeg1D

def export_model():
    print("Đang nạp mô hình đã train...")
    model = QuantResSeg1D(in_channels=1, num_classes=3)
    model.load_state_dict(torch.load("deepdas_brevitas_best.pth", weights_only=True))
    model.eval()

    # Tạo một tensor mẫu đúng với kích thước đầu vào (Batch=1, Kênh=1, Chiều dài=4000)
    input_shape = torch.randn(1, 1, 4000)

    print("Đang xuất ra file QONNX...")
    # FINN yêu cầu tên file kết thúc bằng .qonnx
    bo.export_qonnx(model, input_shape, export_path="deepdas_model.qonnx")
    print("Xuất thành công! Đã sẵn sàng cho Giai đoạn 2 (FINN).")

if __name__ == "__main__":
    export_model()