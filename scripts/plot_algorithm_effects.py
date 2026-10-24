from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from plot_grid import DEFAULT_DPI, DEFAULT_GRID_DIVISIONS, add_raw_pixel_grid, map_title


CHANNELS = ("order", "thermal", "hpwl")


def plot_algorithm_effects(
    input_path: Path,
    output_path: Path,
    labels: list[str],
    *,
    dpi: int = DEFAULT_DPI,
    grid_divisions: int = DEFAULT_GRID_DIVISIONS,
) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        before = np.asarray(data["before_maps"], dtype=np.float32)
        after = np.asarray(data["after_maps"], dtype=np.float32)

    if before.ndim != 3 or before.shape[0] != 3:
        raise ValueError("before_maps must have shape [3, height, width]")
    if after.ndim != 4 or after.shape[1:] != before.shape:
        raise ValueError("after_maps must have shape [algorithm, 3, height, width]")
    if len(labels) != after.shape[0]:
        raise ValueError("one label is required for each algorithm")

    delta = after - before[None, ...]
    fig, axes = plt.subplots(
        len(labels),
        3,
        figsize=(4.8 * 3, 3.8 * len(labels)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, label in enumerate(labels):
        for col, channel in enumerate(CHANNELS):
            axis = axes[row, col]
            image_delta = delta[row, col]
            limit = max(float(np.max(np.abs(image_delta))), 1e-8)
            axis.imshow(
                image_delta,
                cmap="coolwarm",
                origin="lower",
                interpolation="nearest",
                norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            )
            height, width = image_delta.shape
            axis.set_title(
                map_title(
                    f"{label}: Δ {channel}",
                    height=height,
                    width=width,
                    divisions=grid_divisions,
                ),
                fontsize=9,
            )
            add_raw_pixel_grid(
                axis,
                height=height,
                width=width,
                divisions=grid_divisions,
            )

    fig.suptitle("Algorithm-conditioned map changes")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=["sa", "analytical", "rl"],
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--grid-divisions", type=int, default=DEFAULT_GRID_DIVISIONS)
    args = parser.parse_args()
    plot_algorithm_effects(
        args.input,
        args.output,
        args.labels,
        dpi=args.dpi,
        grid_divisions=args.grid_divisions,
    )


if __name__ == "__main__":
    main()
