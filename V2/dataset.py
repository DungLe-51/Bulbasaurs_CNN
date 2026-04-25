import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

class PatchDataset(Dataset):
    def __init__(self, h5_path, win_h, win_w, stride_h, stride_w):
        self.h5_path = h5_path
        self.win_h = win_h
        self.win_w = win_w
        self.stride_h = stride_h
        self.stride_w = stride_w

        with h5py.File(h5_path, "r") as f:
            self.data_shape = f["patch"].shape
            
        H, W = self.data_shape
        self.coords = []
        for i in range(0, H - win_h + 1, stride_h):
            for j in range(0, W - win_w + 1, stride_w):
                self.coords.append((i, j))

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, idx):
        i, j = self.coords[idx]

        with h5py.File(self.h5_path, "r") as f:
            patch = f["patch"][i : i + self.win_h, j : j + self.win_w]
            
            # SỬA LỖI 1: Lấy đủ 3 kênh bằng dấu ':', sau đó mới cắt patch
            mask = f["mask"][:, i : i + self.win_h, j : j + self.win_w] 

        patch = patch.astype(np.float32)
        mask = mask.astype(np.float32)

        # Chuẩn hóa
        patch = (patch - patch.mean(axis=1, keepdims=True)) / (patch.std(axis=1, keepdims=True) + 1e-6)

        # Patch đưa về [1, 4000]
        patch = torch.from_numpy(patch) 
        
        mask = torch.from_numpy(mask)
        # SỬA LỖI 2: Mask lúc này đang là [3, 1, 4000].
        # Bóp bỏ cái chiều cao bằng 1 ở giữa để nó thành [3, 4000] khớp với mạng 1D
        mask = mask.squeeze(1)

        return patch, mask