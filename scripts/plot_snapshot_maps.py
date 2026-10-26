from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from figure_export import export_figure
from plot_grid import (
    FOUR_CHANNEL_LABELS,
    channels_for_count,
    DEFAULT_DPI,
    DEFAULT_GRID_DIVISIONS,
    add_chiplet_outlines,
    add_raw_pixel_grid,
    compute_map_limits,
    configure_colorbar,
    draw_map,
    map_title,
)


def plot_snapshot_maps(
    input_path: Path,
    output_path: Path,
    *,
    dpi: int = DEFAULT_DPI,
    grid_divisions: int = DEFAULT_GRID_DIVISIONS,
    title: str | None = None,
    channel_labels: Sequence[str] | None = None,
    limits: Sequence[tuple[float, float]] | None = None,
    chiplet_boxes: np.ndarray | None = None,
) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        maps = np.asarray(data["maps"])

    if maps.ndim != 3 or maps.shape[0] not in (3, 4):
        raise ValueError("maps must have shape [3 or 4, height, width]")
    if not np.isfinite(maps).all():
        raise ValueError("maps contain NaN or Inf")
    count = maps.shape[0]
    channels = channels_for_count(count)
    channel_labels = FOUR_CHANNEL_LABELS[:count] if channel_labels is None else channel_labels
    if len(channel_labels) != count or (limits is not None and len(limits) != count):
        raise ValueError("channel_labels and limits must match the channel count")
    scales = limits if limits is not None else [
        compute_map_limits(maps[index], channel)
        for index, channel in enumerate(channels)
    ]

    if count == 4:
        legality = maps[3] if maps.ndim == 3 else maps[:, 3]
        compute_map_limits(legality, "legality")
        lo, hi = scales[3]
        if not np.isfinite([lo, hi]).all() or lo != 0 or hi <= lo or hi < np.max(legality):
            raise ValueError("legality limits must start at zero and cover all violations")

    height, width = maps.shape[1:]
    fig, axes = plt.subplots(1, count, figsize=(5 * count, 5), constrained_layout=True)
    for index, (axis, channel, name) in enumerate(zip(axes, channels, channel_labels)):
        image = draw_map(axis, maps[index], channel, limits=scales[index])
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
        if channel == "order" and chiplet_boxes is not None:
            add_chiplet_outlines(axis, chiplet_boxes, height=height, width=width)
        colorbar = fig.colorbar(image, ax=axis, shrink=0.8)
        configure_colorbar(colorbar, image, name)

    if title is not None:
        fig.suptitle(title, fontsize=11)
    try:
        with matplotlib.rc_context({
            "font.family": "sans-serif", "pdf.fonttype": 42, "svg.fonttype": "none",
        }):
            export_figure(fig, output_path, dpi=dpi, axes=axes)
    finally:
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
