#!/usr/bin/env python3
"""
DeepDAS Phase Picking Post-Processing
=====================================

Convert DeepDAS probability maps into P/S phase picks using DBSCAN clustering.
The script can also overlay picks on probability maps or original DAS data.

Example
-------

python run_postprocess.py \
    --input ZI.G20200727_2044.npy \
    --h5_file /home/xiaohan/research/DASdata/cannary/Events_det_TFyGC_sinstacking/ZI.G20200727_2044.h5 \
    --p_threshold 0.8 \
    --s_threshold 0.8 \
    --eps 10 \
    --min_samples 10 \
    --plot_prob \
    --plot_h5 \
    --plot_h5_dir ./

Outputs
-------

phase_picks.npz
    Contains:
        p_centers : P phase picks
        s_centers : S phase picks

Optional figures:
    probability overlay
    DAS waveform overlay
"""

import os
import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt

from preprocess import butter_bandpass_filter, normalize, re_com_mode
from sklearn.cluster import DBSCAN
from scipy.stats import linregress
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# -----------------------------------------------------------
# Utility functions
# -----------------------------------------------------------

def calculate_slope_change(cluster1, cluster2):
    slope1, _, _, _, _ = linregress(cluster1[:, 0], cluster1[:, 1])
    slope2, _, _, _, _ = linregress(cluster2[:, 0], cluster2[:, 1])
    angle = np.arctan(abs((slope2 - slope1) / (1 + slope1 * slope2)))
    return angle


def merge_clusters(X, labels, slope_threshold=0.5, y_diff_threshold=500):
    merged = labels.copy()

    for label in set(labels):
        if label == -1:
            continue

        for other in set(labels):
            if other <= label:
                continue

            if not np.any(merged == label) or not np.any(merged == other):
                continue

            X1 = X[merged == label]
            X2 = X[merged == other]

            slope_change = calculate_slope_change(X1, X2)

            mean1 = np.mean(X1[-100:, 1])
            mean2 = np.mean(X2[-100:, 1])

            if slope_change < slope_threshold and abs(mean2 - mean1) < y_diff_threshold:
                merged[merged == other] = label

    return merged


def remove_outliers_polynomial(X, labels, degree=3, threshold_factor=3):
    new_labels = labels.copy()

    poly = PolynomialFeatures(degree=degree)

    for label in set(labels):

        if label == -1:
            continue

        cluster_points = X[new_labels == label]

        if len(cluster_points) < 3:
            continue

        X_feat = cluster_points[:, 0].reshape(-1, 1)
        y = cluster_points[:, 1]

        X_poly = poly.fit_transform(X_feat)

        model = LinearRegression()
        model.fit(X_poly, y)

        y_pred = model.predict(X_poly)

        residuals = y - y_pred

        std = np.std(residuals)

        mask = np.abs(residuals) > threshold_factor * std

        indices = np.where(new_labels == label)[0][mask]

        new_labels[indices] = -1

    return new_labels


# -----------------------------------------------------------
# Cluster extraction
# -----------------------------------------------------------

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

            cluster_points = coords[labels == l]

            center = cluster_points.mean(axis=0)

            centers.append(center)

    return np.array(centers)


# -----------------------------------------------------------
# Plot probability overlay
# -----------------------------------------------------------

def plot_overlay_prob(h5_file, prob_maps, p_centers, s_centers, save_path=None):

    plt.figure(figsize=(12, 6))

    plt.imshow(1-prob_maps[2], aspect="auto", cmap="viridis", vmin=0, vmax=1)

    if len(p_centers) > 0:
        plt.scatter(p_centers[:, 1], p_centers[:, 0],
                    c="red", s=0.3, label="P picks")

    if len(s_centers) > 0:
        plt.scatter(s_centers[:, 1], s_centers[:, 0],
                    c="blue", s=0.3, label="S picks")

    plt.xlabel("Time Index")
    plt.ylabel("Channel")
    plt.title("Phase Picks Overlay on Probability Map")
    plt.legend()
    plt.tight_layout()
    
    name = os.path.splitext(os.path.basename(args.h5_file))[0]
    plt.savefig(f"{name}_phase_probability_overlay.png", dpi=300)
    print("Saved probability overlay:", name)

    plt.show()


# -----------------------------------------------------------
# Plot DAS overlay
# -----------------------------------------------------------

def plot_overlay_h5(h5_file, p_centers, s_centers, dataset_name="data", save_dir="./"):

    with h5py.File(h5_file, "r") as f:

        if dataset_name not in f:
            raise ValueError(
                f"Dataset '{dataset_name}' not found. Available: {list(f.keys())}"
            )

        data = f[dataset_name][...]
        if args.data_order == "time_channel":
            data = data.T  # convert to (channel, time)
            
    abs_median = np.median(np.abs(data))
    vmin = -0.5 * abs_median
    vmax = 0.5 * abs_median
    
    plt.figure(figsize=(12, 6))
    plt.imshow(re_com_mode(data), aspect="auto", cmap="seismic", vmin=vmin, vmax=vmax)

    if len(p_centers) > 0:
        plt.scatter(p_centers[:, 1], p_centers[:, 0],
                    c="red", s=0.3, label="P picks")

    if len(s_centers) > 0:
        plt.scatter(s_centers[:, 1], s_centers[:, 0],
                    c="blue", s=0.3, label="S picks")

    plt.xlabel("Time Index")
    plt.ylabel("Channel")
    plt.title("Phase Picks Overlay on DAS Data")

    #plt.colorbar(label="Amplitude")
    plt.legend()

    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)

    name = os.path.splitext(os.path.basename(h5_file))[0]

    out_file = os.path.join(save_dir, f"{name}_phase_overlay.png")

    plt.savefig(out_file, dpi=300)

    print("Saved DAS overlay:", out_file)

    plt.show()


# -----------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------

def main(args):

    print("Loading probability maps...")
    combined_output = np.load(args.input)

    print("Extracting P clusters...")
    p_centers = extract_clusters(
        combined_output[0],
        threshold=args.p_threshold,
        eps=args.eps,
        min_samples=args.min_samples
    )

    print("Extracting S clusters...")
    s_centers = extract_clusters(
        combined_output[1],
        threshold=args.s_threshold,
        eps=args.eps,
        min_samples=args.min_samples
    )

    print("Detected P picks:", len(p_centers))
    print("Detected S picks:", len(s_centers))
    
    if args.remove_outliers:

        print("Removing outliers...")

        p_labels = np.zeros(len(p_centers))
        s_labels = np.zeros(len(s_centers))

        p_labels = remove_outliers_polynomial(p_centers, p_labels)
        s_labels = remove_outliers_polynomial(s_centers, s_labels)

        p_centers = p_centers[p_labels != -1]
        s_centers = s_centers[s_labels != -1]
    name = os.path.splitext(os.path.basename(args.h5_file))[0]
    output_file = name + ".npz"
    np.savez(output_file, p_centers=p_centers, s_centers=s_centers)
    print("Saved phase picks:", output_file)

    if args.plot_prob:

        plot_overlay_prob(
            args.h5_file,
            combined_output,
            p_centers,
            s_centers,
            save_path=args.plot_prob_file
        )

    if args.plot_h5 and args.h5_file:

        plot_overlay_h5(
            args.h5_file,
            p_centers,
            s_centers,
            dataset_name=args.dataset_name,
            save_dir=args.plot_h5_dir
        )


# -----------------------------------------------------------
# CLI
# -----------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="DeepDAS phase pick post-processing")
    parser.add_argument("--input", required=True, help="Input probability map (.npy)")
    parser.add_argument("--h5_file", help="Original DAS HDF5 file")
    parser.add_argument("--dataset_name", default="data", help="Dataset name inside HDF5")
    parser.add_argument("--data_order", type=str, default="channel_time",
                        choices=["channel_time", "time_channel"],
                        help="Dataset shape: 'channel_time'=(channel,time), 'time_channel'=(time,channel). "
                             "If 'time_channel', data will be transposed automatically.")
    parser.add_argument("--p_threshold", type=float, default=0.7, help="P-phase probability threshold")
    parser.add_argument("--s_threshold", type=float, default=0.7, help="S-phase probability threshold")
    parser.add_argument("--eps", type=float, default=5, help="DBSCAN eps parameter")
    parser.add_argument("--min_samples", type=int, default=5, help="DBSCAN minimum samples")
    parser.add_argument("--remove_outliers", action="store_true", help="Remove outlier picks after clustering")
    parser.add_argument("--plot_prob", action="store_true", help="Plot probability overlay figure")
    parser.add_argument("--plot_prob_file", default="phase_prob_overlay.png", help="Probability plot output file")
    parser.add_argument("--plot_h5", action="store_true", help="Plot DAS waveform with picks")
    parser.add_argument("--plot_h5_dir", default="./", help="Directory to save H5 plots")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)