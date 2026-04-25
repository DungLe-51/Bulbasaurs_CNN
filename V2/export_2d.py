import torch
import torch.nn as nn
import brevitas.nn as qnn
import brevitas.onnx as bo

# 1. Khai báo phiên bản 2D "Ảo" của mạng (Height = 1)
class QuantResBlock2D(nn.Module):
    def __init__(self, c, k=3, weight_bit=8, act_bit=8):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            # Thay Conv1d bằng Conv2d, kernel (1, k), padding (0, p)
            qnn.QuantConv2d(c, c, (1, k), padding=(0, p), weight_bit_width=weight_bit, bias=False),
            nn.BatchNorm2d(c),
            qnn.QuantReLU(bit_width=act_bit),
            nn.Dropout(0.1),
            qnn.QuantConv2d(c, c, (1, k), padding=(0, p), weight_bit_width=weight_bit, bias=False),
            nn.BatchNorm2d(c),
        )
        self.act = qnn.QuantReLU(bit_width=act_bit)

    def forward(self, x):
        return self.act(x + self.block(x))

class QuantResSeg2D(nn.Module):
    def __init__(self, in_channels=1, base=64, blocks=4, num_classes=3, weight_bit=8, act_bit=8):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=act_bit)
        self.stem = nn.Sequential(
            qnn.QuantConv2d(in_channels, base, kernel_size=(1, 7), padding=(0, 3), weight_bit_width=weight_bit, bias=False),
            nn.BatchNorm2d(base),
            qnn.QuantReLU(bit_width=act_bit)
        )
        self.res = nn.Sequential(*[QuantResBlock2D(base, k=3, weight_bit=weight_bit, act_bit=act_bit) for _ in range(blocks)])
        self.head = qnn.QuantConv2d(base, num_classes, kernel_size=(1, 1), weight_bit_width=weight_bit, bias=False)

    def forward(self, x):
        x = self.quant_inp(x)
        h = self.res(self.stem(x))
        out = self.head(h)
        return out

def main():
    print("1. Khởi tạo mô hình 2D ảo...")
    model_2d = QuantResSeg2D(in_channels=1, num_classes=3)

    print("2. Tải trọng số 1D và ép sang Tensor 2D...")
    # Sửa tên file .pth cho đúng với file trọng số tốt nhất bạn đang có
    state_dict_1d = torch.load("deepdas_brevitas_best.pth", map_location="cpu", weights_only=True)
    state_dict_2d = {}
    
    for key, tensor in state_dict_1d.items():
        # Trọng số Conv1d là [Out, In, Length]. Bơm thêm 1 chiều để thành [Out, In, 1, Length]
        if len(tensor.shape) == 3:
            state_dict_2d[key] = tensor.unsqueeze(2)
        else:
            state_dict_2d[key] = tensor

    model_2d.load_state_dict(state_dict_2d)
    model_2d.eval()

    print("3. Xuất file QONNX...")
    # Tensor đầu vào giờ là 4 chiều: [Batch=1, Channel=1, Height=1, Width=4000]
    input_shape = torch.randn(1, 1, 1, 4000)
    bo.export_qonnx(model_2d, input_shape, export_path="deepdas_model_ready.qonnx")
    print("HOÀN TẤT! Đã tạo deepdas_model_ready.qonnx chuẩn FINN!")

if __name__ == "__main__":
    main()