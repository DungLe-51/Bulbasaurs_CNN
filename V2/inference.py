import numpy as np
import torch

def sliding_window_indices(H, W, win_h, win_w, stride_h, stride_w):
    indices = []
    for i in range(0, H, stride_h):
        for j in range(0, W, stride_w):
            end_i = min(i + win_h, H)
            end_j = min(j + win_w, W)
            start_i = max(end_i - win_h, 0)
            start_j = max(end_j - win_w, 0)
            indices.append((start_i, start_j))
        if i + win_h >= H:
            break
    return list(set(indices))

def build_weight_window(h, w):
    hann_h = np.hanning(h)
    hann_w = np.hanning(w)
    return np.outer(hann_h, hann_w) + 1e-6

def infer_full_data(model, full_data, win_h, win_w, stride_h, stride_w, device, batch_size=4, use_amp=False):
    H, W = full_data.shape
    indices = sliding_window_indices(H, W, win_h, win_w, stride_h, stride_w)
    combined_output = np.zeros((3, H, W))
    weight_map = np.zeros((3, H, W))
    weight_window = build_weight_window(win_h, win_w)

    batch_patches = []
    batch_positions = []

    for idx, (i, j) in enumerate(indices):
        patch = full_data[i:i+win_h, j:j+win_w]
        patch_tensor = torch.tensor(patch).unsqueeze(0).unsqueeze(0).float()
        batch_patches.append(patch_tensor)
        batch_positions.append((i, j))

        if len(batch_patches) == batch_size or idx == len(indices)-1:
            batch_tensor = torch.cat(batch_patches, dim=0).to(device)
            with torch.no_grad():
                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_tensor)['out']
                else:
                    outputs = model(batch_tensor)['out']
            outputs = outputs.cpu().numpy()
            for b in range(len(batch_positions)):
                i0, j0 = batch_positions[b]
                out_patch = outputs[b]
                for c in range(3):
                    combined_output[c,
                                    i0:i0+win_h,
                                    j0:j0+win_w] += out_patch[c] * weight_window
                    weight_map[c,
                               i0:i0+win_h,
                               j0:j0+win_w] += weight_window
            batch_patches = []
            batch_positions = []

    combined_output = combined_output / (weight_map + 1e-8)
    return combined_output