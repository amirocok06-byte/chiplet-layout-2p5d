from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ("order", "thermal", "hpwl")
CMAPS = ("viridis", "inferno", "magma")


def plot_snapshot_maps(input_path: Path, output_path: Path) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        maps = np.asarray(data["maps"], dtype=np.float32)

    if maps.ndim != 3 or maps.shape[0] != 3:
        raise ValueError("maps must have shape [3, height, width]")
    if not np.isfinite(maps).all():
        raise ValueError("maps contain NaN or Inf")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    for index, (axis, name, cmap) in enumerate(zip(axes, CHANNELS, CMAPS)):
        image = axis.imshow(
            maps[index],
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
        )
        axis.set_title(name)
        axis.set_xticks([])
        axis.set_yticks([])
        fig.colorbar(image, ax=axis, shrink=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    plot_snapshot_maps(args.input, args.output)


if __name__ == "__main__":
    main()
