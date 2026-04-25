import matplotlib.pyplot as plt
import os

def plot_combined_output(combined_output, output_prefix, vmin=0, vmax=1, cmap='viridis'):
    channels = combined_output.shape[0]
    for c in range(channels):
        plt.rcParams.update({'font.size': 15})
        plt.figure(figsize=(12, 6))
        plt.imshow(combined_output[c], aspect='auto', vmin=vmin, vmax=vmax, cmap=cmap)
        plt.colorbar()
        plt.title(f"Channel {c+1} prediction")
        plt.xlabel("Time (samples)")
        plt.ylabel("Channel Number")
        plt.tight_layout()
        save_path = f"{output_prefix}_channel{c+1}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Saved plot: {save_path}")