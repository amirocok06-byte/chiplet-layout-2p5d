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
    CHANNEL_LABELS,
    CHANNELS,
    DEFAULT_DPI,
    DEFAULT_GRID_DIVISIONS,
    add_chiplet_outlines,
    add_raw_pixel_grid,
    compute_map_limits,
    configure_colorbar,
    draw_map,
    map_title,
)


def plot_state_flow(
    input_path: Path,
    output_path: Path,
    max_frames: int = 8,
    *,
    dpi: int = DEFAULT_DPI,
    grid_divisions: int = DEFAULT_GRID_DIVISIONS,
    frame_labels: list[str] | None = None,
    title: str | None = "State flow: order / thermal / HPWL",
    channel_labels: tuple[str, str, str] = CHANNEL_LABELS,
    limits: Sequence[tuple[float, float]] | None = None,
    chiplet_boxes: Sequence[np.ndarray] | None = None,
) -> None:
    with np.load(input_path, allow_pickle=False) as data:
        maps = np.asarray(data["maps"])

    if maps.ndim != 4 or maps.shape[1] != 3:
        raise ValueError("maps must have shape [time, 3, height, width]")
    if not np.isfinite(maps).all():
        raise ValueError("maps contain NaN or Inf")
    if maps.shape[0] == 0 or max_frames <= 0:
        raise ValueError("state flow and max_frames must contain at least one frame")
    if frame_labels is not None and len(frame_labels) != maps.shape[0]:
        raise ValueError("one frame label is required for each input frame")
    if chiplet_boxes is not None and len(chiplet_boxes) != maps.shape[0]:
        raise ValueError("chiplet_boxes must contain one box array per input frame")
    if len(channel_labels) != 3 or (limits is not None and len(limits) != 3):
        raise ValueError("channel_labels and limits must each contain three entries")
    # Include undisplayed frames in every scale, even when max_frames samples T.
    scales = limits if limits is not None else [
        compute_map_limits(maps[:, row], channel)
        for row, channel in enumerate(CHANNELS)
    ]

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

    for row, (channel, name) in enumerate(zip(CHANNELS, channel_labels)):
        reference_image = None
        for col, frame_id in enumerate(frame_ids):
            axis = axes[row, col]
            image = draw_map(axis, maps[frame_id, row], channel, limits=scales[row])
            if reference_image is None:
                reference_image = image
            else:
                image.set_norm(reference_image.norm)
            height, width = maps.shape[2:]
            frame_label = f"t={frame_id}" if frame_labels is None else frame_labels[frame_id]
            axis.set_title(
                map_title(
                    f"{name}, {frame_label}",
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
            if channel == "order" and chiplet_boxes is not None:
                add_chiplet_outlines(axis, chiplet_boxes[frame_id], height=height, width=width)
        colorbar = fig.colorbar(reference_image, ax=axes[row, :].tolist(), shrink=0.8)
        configure_colorbar(colorbar, reference_image, name)

    if title is not None:
        fig.suptitle(title, fontsize=11)
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
