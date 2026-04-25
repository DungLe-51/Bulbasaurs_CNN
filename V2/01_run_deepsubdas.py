#!/usr/bin/env python3
"""
DeepDAS HPC Inference Script
-----------------------------------
Perform patch-based inference on DAS HDF5 data using a trained DeepDAS model.
Automatically saves combined output and plots.

Example usage:
python run_deepsubdas.py \
    --model_path 2025_deepdas_1.0.pth \
    --input_h5 ZI.G20200727_2044.h5 \
    --data_order channel_time \
    --dataset_name data \
    --channel_w 2000 \
    --time_w 4000 \
    --stride_c 1000 \
    --stride_t 2000 \
    --batch_size 1 \
    --use_amp
"""

import argparse
import os
import h5py
import numpy as np
import torch

from models import load_model
from preprocess import butter_bandpass_filter, normalize, re_com_mode
from inference import infer_full_data
from plot_utils import plot_combined_output


def parse_args():
    parser = argparse.ArgumentParser(
        description="DeepDAS HPC Inference",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--model_path", type=str, required=True, help="Path to trained DeepDAS model (.pth)")
    parser.add_argument("--input_h5", type=str, required=True, help="Path to DAS HDF5 input file")
    parser.add_argument("--dataset_name", type=str, default="data", help="Dataset name inside HDF5 (default: 'data')")
    parser.add_argument("--data_order", type=str,default="channel_time", choices=["channel_time", "time_channel"],
    help="Shape of the dataset inside HDF5. "
         "'channel_time' means (channel, time). "
         "'time_channel' means (time, channel). "
         "If 'time_channel' is used, the data will be automatically transposed.")
    parser.add_argument("--channel_w", type=int, default=1000, help="Patch height (channels)")
    parser.add_argument("--time_w", type=int, default=2000, help="Patch width (time samples)")
    parser.add_argument("--stride_c", type=int, default=None, help="Channel stride (default: half patch height)")
    parser.add_argument("--stride_t", type=int, default=None, help="Time stride (default: half patch width)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for inference")
    parser.add_argument("--use_amp", action="store_true", help="Use mixed precision (FP16) for faster inference")
    parser.add_argument("--vmin", type=float, default=0, help="vmin for plotting (default: 0)")
    parser.add_argument("--vmax", type=float, default=1, help="vmax for plotting (default: 1)")
    parser.add_argument("--cmap", type=str, default="viridis", help="Colormap for plotting")

    return parser.parse_args()


def main():
    args = parse_args()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load model
    print("Loading model from:", args.model_path)
    model = load_model(args.model_path, device)

    # Load DAS data
    print("Loading DAS data from:", args.input_h5)
    with h5py.File(args.input_h5, "r") as f:
        if args.dataset_name not in f:
            raise ValueError(f"Dataset '{args.dataset_name}' not found in {args.input_h5}. Available datasets: {list(f.keys())}")
        full_data = f[args.dataset_name][...]
        if args.data_order == "time_channel":
            full_data = full_data.T  # convert to (channel, time)
        print(f"Loaded dataset '{args.dataset_name}' with shape {full_data.shape}")

    # Preprocessing
    print("Preprocessing data...")
    full_data = butter_bandpass_filter(full_data)
    full_data = normalize(full_data)
    full_data = re_com_mode(full_data)

    # Determine strides
    H, W = full_data.shape
    win_h = args.channel_w
    win_w = args.time_w
    stride_h = args.stride_c if args.stride_c else win_h // 2
    stride_w = args.stride_t if args.stride_t else win_w // 2
    print(f"Patch size: ({win_h}, {win_w}), stride: ({stride_h}, {stride_w})")

    # Run inference
    print("Running inference...")
    combined_output = infer_full_data(
        model, full_data, win_h, win_w, stride_h, stride_w,
        device=device, batch_size=args.batch_size, use_amp=args.use_amp
    )

    # Save combined output
    input_basename = os.path.basename(args.input_h5)
    output_prefix = os.path.splitext(input_basename)[0]
    output_npy = output_prefix + ".npy"
    np.save(output_npy, combined_output)
    print(f"Saved combined output: {output_npy}")

    # Plot combined output
    plot_combined_output(
        combined_output,
        output_prefix,
        vmin=args.vmin,
        vmax=args.vmax,
        cmap=args.cmap
    )
    print("Plots saved.")


if __name__ == "__main__":
    main()