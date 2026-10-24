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


def plot_state_flow(
    input_path: Path,
    output_path: Path,
    max_frames: int = 8,
    *,
    dpi: int = DEFAULT_DPI,
    grid_divisions: int = DEFAULT_GRID_DIVISIONS,
) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        maps = np.asarray(data["maps"], dtype=np.float32)

    if maps.ndim != 4 or maps.shape[1] != 3:
        raise ValueError("maps must have shape [time, 3, height, width]")
    if not np.isfinite(maps).all():
        raise ValueError("maps contain NaN or Inf")

    frame_ids = np.unique(
        np.linspace(0, maps.shape[0] - 1, min(max_frames, maps.shape[0])).astype(int)
    )
    fig, axes = plt.subplots(
        3,
        len(frame_ids),
        figsize=(3.6 * len(frame_ids), 8.5),
        squeeze=False,
        constrained_layout=True,
    )

    for row, (name, cmap) in enumerate(zip(CHANNELS, CMAPS)):
        lo = float(np.nanmin(maps[:, row]))
        hi = float(np.nanmax(maps[:, row]))
        for col, frame_id in enumerate(frame_ids):
            axis = axes[row, col]
            axis.imshow(
                maps[frame_id, row],
                cmap=cmap,
                origin="lower",
                interpolation="nearest",
                vmin=lo,
                vmax=hi if hi > lo else lo + 1.0,
            )
            height, width = maps.shape[2:]
            axis.set_title(
                map_title(
                    f"{name}, t={frame_id}",
                    height=height,
                    width=width,
                    divisions=grid_divisions,
                ),
                fontsize=8,
            )
            add_raw_pixel_grid(
                axis,
                height=height,
                width=width,
                divisions=grid_divisions,
            )

    fig.suptitle("State flow: order / thermal / HPWL")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-frames", type=int, default=8)
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--grid-divisions", type=int, default=DEFAULT_GRID_DIVISIONS)
    args = parser.parse_args()
    plot_state_flow(
        args.input,
        args.output,
        max_frames=args.max_frames,
        dpi=args.dpi,
        grid_divisions=args.grid_divisions,
    )


if __name__ == "__main__":
    main()
