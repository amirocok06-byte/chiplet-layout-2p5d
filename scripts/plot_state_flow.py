from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


CHANNELS = ("order", "thermal", "hpwl")
CMAPS = ("viridis", "inferno", "magma")


def plot_state_flow(input_path: Path, output_path: Path, max_frames: int = 8) -> None:
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
        figsize=(2.6 * len(frame_ids), 7.5),
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
            axis.set_title(f"{name}, t={frame_id}", fontsize=8)
            axis.set_xticks([])
            axis.set_yticks([])

    fig.suptitle("State flow: order / thermal / HPWL")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--max-frames", type=int, default=8)
    args = parser.parse_args()
    plot_state_flow(args.input, args.output, max_frames=args.max_frames)


if __name__ == "__main__":
    main()
