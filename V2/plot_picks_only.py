import argparse
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_picks(p_centers, s_centers):

    plt.figure(figsize=(10, 6))

    if p_centers is not None and len(p_centers) > 0:
        plt.scatter(p_centers[:, 1], p_centers[:, 0], c="red", s=20, label="P picks")

    if s_centers is not None and len(s_centers) > 0:
        plt.scatter(s_centers[:, 1], s_centers[:, 0], c="blue", s=20, label="S picks")

    plt.xlabel("Time Index")
    plt.ylabel("Channel")
    plt.title("Phase Picks")
    plt.gca().invert_yaxis()
    plt.legend()
    plt.tight_layout()
    name = os.path.splitext(os.path.basename(args.input))[0]
    out_file = f"{name}_picks.png"
    plt.savefig(out_file, dpi=300)
    plt.show()


def main(args):

    print("Loading:", args.input)
    data = np.load(args.input)

    p_centers = data["p_centers"]
    s_centers = data["s_centers"]

    print("P picks:", len(p_centers))
    print("S picks:", len(s_centers))

    plot_picks(p_centers, s_centers)


def parse_args():

    parser = argparse.ArgumentParser(
        description="Plot P and S phase picks from npz file"
    )

    parser.add_argument(
        "--input",
        required=True,
        help="Input npz file containing p_centers and s_centers"
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)