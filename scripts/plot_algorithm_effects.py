from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


CHANNELS = ("order", "thermal", "hpwl")


def plot_algorithm_effects(
    input_path: Path,
    output_path: Path,
    labels: list[str],
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
        figsize=(4 * 3, 3.3 * len(labels)),
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
            axis.set_title(f"{label}: delta {channel}")
            axis.set_xticks([])
            axis.set_yticks([])

    fig.suptitle("Algorithm-conditioned map changes")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
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
    args = parser.parse_args()
    plot_algorithm_effects(args.input, args.output, args.labels)


if __name__ == "__main__":
    main()
