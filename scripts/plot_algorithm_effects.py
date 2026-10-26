from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from figure_export import export_figure
from plot_grid import (
    channels_for_count,
    DEFAULT_DPI,
    DEFAULT_GRID_DIVISIONS,
    add_raw_pixel_grid,
    configure_colorbar,
    map_title,
)


def plot_algorithm_effects(
    input_path: Path,
    output_path: Path,
    labels: list[str],
    *,
    dpi: int = DEFAULT_DPI,
    grid_divisions: int = DEFAULT_GRID_DIVISIONS,
    limits: Sequence[tuple[float, float]] | None = None,
) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        before = np.asarray(data["before_maps"])
        after = np.asarray(data["after_maps"])

    if before.ndim != 3 or before.shape[0] not in (3, 4):
        raise ValueError("before_maps must have shape [3 or 4, height, width]")
    if after.ndim != 4 or after.shape[1:] != before.shape:
        raise ValueError("after_maps must have shape [algorithm, 3 or 4, height, width]")
    if len(labels) != after.shape[0]:
        raise ValueError("one label is required for each algorithm")
    if not labels:
        raise ValueError("at least one algorithm is required")
    if not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("maps contain NaN or Inf")

    count = before.shape[0]
    channels = channels_for_count(count)
    if limits is not None and len(limits) != count:
        raise ValueError("limits must match the channel count")
    display_names = ("Connectivity priority", "Spatial thermal proxy",
                     "HPWL contribution", "Overlap / boundary violation")
    delta = after.astype(np.float64) - before[None, ...]
    # One symmetric range per channel across every algorithm, including zeros.
    norms = [
        TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        for limit in np.maximum(np.max(np.abs(delta), axis=(0, 2, 3)), 1e-8)
    ]
    fig, axes = plt.subplots(
        len(labels),
        count,
        figsize=(4.8 * count, 3.8 * len(labels)),
        squeeze=False,
        constrained_layout=True,
    )

    for row, label in enumerate(labels):
        for col, channel in enumerate(channels):
            axis = axes[row, col]
            image_delta = delta[row, col]
            axis.imshow(
                    image_delta,
                    cmap="coolwarm",
                    origin="lower",
                    interpolation="nearest",
                    norm=(TwoSlopeNorm(vmin=limits[col][0], vcenter=0., vmax=limits[col][1])
                          if limits is not None else norms[col]),
            )
            height, width = image_delta.shape
            axis.set_title(
                map_title(
                    f"{label}: Δ {display_names[col]}",
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
    for col, name in enumerate(display_names[:count]):
        colorbar = fig.colorbar(axes[0, col].images[0], ax=axes[:, col].tolist(), shrink=0.8)
        configure_colorbar(colorbar, axes[0, col].images[0], f"Δ {name}")

    fig.suptitle("Algorithm-conditioned map changes", fontsize=11)
    try:
        with matplotlib.rc_context({
            "font.family": "sans-serif", "pdf.fonttype": 42, "svg.fonttype": "none",
        }):
            export_figure(fig, output_path, dpi=dpi, axes=axes.ravel())
    finally:
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
