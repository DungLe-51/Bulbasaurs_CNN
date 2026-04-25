#!/usr/bin/env python3
"""
DeepDAS Batch Inference + Phase Picking
=======================================

Run DeepDAS inference and phase picking directly on HDF5 files.
Outputs only overlay plots.

Example
-------

python run_deepdas_batch.py \
    --model_path 2025_deepdas_1.0.pth \
    --input_h5 ZI.G20200727_*.h5 \
    --dataset_name data \
    --channel_w 2000 \
    --time_w 4000 \
    --batch_size 1 \
    --p_threshold 0.8 \
    --s_threshold 0.8
"""

import argparse
import os
import glob
import h5py
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN

from models import load_model
from preprocess import butter_bandpass_filter, normalize, re_com_mode
from inference import infer_full_data


# ------------------------------------------------------------
# Cluster extraction
# ------------------------------------------------------------

def extract_clusters(prob_map, threshold=0.7, eps=5, min_samples=5):

    centers = []

    for row in range(prob_map.shape[0]):

        cols = np.where(prob_map[row] > threshold)[0]

        coords = np.column_stack((np.full_like(cols, row), cols))

        if len(coords) < 2:
            continue

        db = DBSCAN(eps=eps, min_samples=min_samples)
        labels = db.fit_predict(coords)

        mask = labels != -1

        coords = coords[mask]
        labels = labels[mask]

        for l in np.unique(labels):
            cluster = coords[labels == l]
            centers.append(cluster.mean(axis=0))

    return np.array(centers)


# ------------------------------------------------------------
# Plot overlay
# ------------------------------------------------------------

def plot_overlay(data, p_centers, s_centers, save_path):

    abs_median = np.median(np.abs(data))
    vmin = -0.5 * abs_median
    vmax = 0.5 * abs_median

    plt.figure(figsize=(12,6))

    plt.imshow(data, aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax)

    if len(p_centers) > 0:
        plt.scatter(p_centers[:,1], p_centers[:,0],
                    c="red", s=0.5, label="P")

    if len(s_centers) > 0:
        plt.scatter(s_centers[:,1], s_centers[:,0],
                    c="blue", s=0.5, label="S")

    plt.xlabel("Time")
    plt.ylabel("Channel")
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print("Saved:", save_path)


# ------------------------------------------------------------
# Process single file
# ------------------------------------------------------------

def process_file(h5_file, args, model, device):

    print("\nProcessing:", h5_file)

    with h5py.File(h5_file,"r") as f:

        data = f[args.dataset_name][...]

        if args.data_order == "time_channel":
            data = data.T

    # preprocess
    data = butter_bandpass_filter(data)
    data = normalize(data)
    data = re_com_mode(data)

    H,W = data.shape

    win_h = args.channel_w
    win_w = args.time_w

    stride_h = args.stride_c or win_h//2
    stride_w = args.stride_t or win_w//2

    print("Running inference...")

    prob = infer_full_data(
        model,
        data,
        win_h,
        win_w,
        stride_h,
        stride_w,
        device=device,
        batch_size=args.batch_size,
        use_amp=args.use_amp
    )

    print("Picking phases...")

    p_centers = extract_clusters(
        prob[0],
        threshold=args.p_threshold,
        eps=args.eps,
        min_samples=args.min_samples
    )

    s_centers = extract_clusters(
        prob[1],
        threshold=args.s_threshold,
        eps=args.eps,
        min_samples=args.min_samples
    )

    name = os.path.splitext(os.path.basename(h5_file))[0]
    out = name + "_phase_overlay.png"

    plot_overlay(data, p_centers, s_centers, out)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--model_path", required=True)

    parser.add_argument("--input_h5", required=True,
                        help="H5 file or wildcard (e.g. data/*.h5)")

    parser.add_argument("--dataset_name", default="data")

    parser.add_argument("--data_order", default="channel_time",
                        choices=["channel_time","time_channel"])

    parser.add_argument("--channel_w", type=int, default=2000)
    parser.add_argument("--time_w", type=int, default=4000)

    parser.add_argument("--stride_c", type=int)
    parser.add_argument("--stride_t", type=int)

    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true")

    parser.add_argument("--p_threshold", type=float, default=0.8)
    parser.add_argument("--s_threshold", type=float, default=0.8)

    parser.add_argument("--eps", type=float, default=10)
    parser.add_argument("--min_samples", type=int, default=10)

    return parser.parse_args()


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():

    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Using device:", device)

    print("Loading model:", args.model_path)

    model = load_model(args.model_path, device)

    files = sorted(glob.glob(args.input_h5))

    print("Found",len(files),"files")

    for f in files:

        process_file(f, args, model, device)


if __name__ == "__main__":
    main()