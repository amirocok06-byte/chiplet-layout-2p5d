from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


DEFAULT_DPI = 300
DEFAULT_GRID_DIVISIONS = 16


def add_raw_pixel_grid(
    axis: plt.Axes,
    *,
    height: int,
    width: int,
    divisions: int = DEFAULT_GRID_DIVISIONS,
) -> None:
    """Annotate a raw map without interpolating or resampling its pixels.

    The grid is a display aid: for the standard 256x256 maps, each of the
    16x16 macro-cells spans exactly 16x16 raw pixels. Boundary labels use raw
    pixel coordinates, including the right/top extent at 256.
    """
    if divisions <= 0:
        raise ValueError("divisions must be positive")
    divisions = min(divisions, height, width)
    x_boundaries = np.linspace(-0.5, width - 0.5, divisions + 1)
    y_boundaries = np.linspace(-0.5, height - 0.5, divisions + 1)
    x_labels = np.rint(np.linspace(0, width, divisions + 1)).astype(int)
    y_labels = np.rint(np.linspace(0, height, divisions + 1)).astype(int)

    axis.set_xticks(x_boundaries)
    axis.set_yticks(y_boundaries)
    axis.set_xticklabels(x_labels, fontsize=6)
    axis.set_yticklabels(y_labels, fontsize=6)
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(-0.5, height - 0.5)
    axis.set_xlabel("raw x (px)", fontsize=7)
    axis.set_ylabel("raw y (px)", fontsize=7)
    axis.grid(True, color="white", alpha=0.38, linewidth=0.45)
    axis.set_aspect("equal", adjustable="box")


def map_title(name: str, *, height: int, width: int, divisions: int) -> str:
    return f"{name}\nraw {width}×{height}; display grid {divisions}×{divisions}"
