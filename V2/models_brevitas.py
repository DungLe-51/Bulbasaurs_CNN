import torch
import torch.nn as nn
import brevitas.nn as qnn

class QuantResBlock1D(nn.Module):
    def __init__(self, c, k=3, weight_bit=8, act_bit=8):
        super().__init__()
        p = k // 2
        self.block = nn.Sequential(
            qnn.QuantConv1d(c, c, k, padding=p, weight_bit_width=weight_bit, bias=False),
            nn.BatchNorm1d(c),
            qnn.QuantReLU(bit_width=act_bit),
            nn.Dropout(0.1),
            qnn.QuantConv1d(c, c, k, padding=p, weight_bit_width=weight_bit, bias=False),
            nn.BatchNorm1d(c),
        )
        self.act = qnn.QuantReLU(bit_width=act_bit)

    def forward(self, x):
        return self.act(x + self.block(x))

class QuantResSeg1D(nn.Module):
    def __init__(self, in_channels=1, base=64, blocks=4, num_classes=3, weight_bit=8, act_bit=8):
        super().__init__()
        self.quant_inp = qnn.QuantIdentity(bit_width=act_bit)
        self.stem = nn.Sequential(
            qnn.QuantConv1d(in_channels, base, kernel_size=7, padding=3, weight_bit_width=weight_bit, bias=False),
            nn.BatchNorm1d(base),
            qnn.QuantReLU(bit_width=act_bit)
        )
        self.res = nn.Sequential(*[QuantResBlock1D(base, k=3, weight_bit=weight_bit, act_bit=act_bit) for _ in range(blocks)])
        self.head = qnn.QuantConv1d(base, num_classes, kernel_size=1, weight_bit_width=weight_bit, bias=False)

    def forward(self, x):
        x = self.quant_inp(x)
        h = self.res(self.stem(x))
        out = self.head(h)
        return out