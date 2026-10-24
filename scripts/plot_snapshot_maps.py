from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from plot_grid import DEFAULT_DPI, DEFAULT_GRID_DIVISIONS, add_raw_pixel_grid, map_title


CHANNELS = ("order", "thermal", "hpwl")
CMAPS = ("viridis", "inferno", "magma")


def plot_snapshot_maps(
    input_path: Path,
    output_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    grid_divisions: int = DEFAULT_GRID_DIVISIONS,
) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        maps = np.asarray(data["maps"], dtype=np.float32)

    if maps.ndim != 3 or maps.shape[0] != 3:
        raise ValueError("maps must have shape [3, height, width]")
    if not np.isfinite(maps).all():
        raise ValueError("maps contain NaN or Inf")

    height, width = maps.shape[1:]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for index, (axis, name, cmap) in enumerate(zip(axes, CHANNELS, CMAPS)):
        image = axis.imshow(
            maps[index],
            cmap=cmap,
            origin="lower",
            interpolation="nearest",
        )
        axis.set_title(
            map_title(name, height=height, width=width, divisions=grid_divisions),
            fontsize=9,
        )
        add_raw_pixel_grid(
            axis,
            height=height,
            width=width,
            divisions=grid_divisions,
        )
        fig.colorbar(image, ax=axis, shrink=0.8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--grid-divisions", type=int, default=DEFAULT_GRID_DIVISIONS)
    args = parser.parse_args()
    plot_snapshot_maps(
        args.input,
        args.output,
        dpi=args.dpi,
        grid_divisions=args.grid_divisions,
    )


if __name__ == "__main__":
    main()
