from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize, PowerNorm, SymLogNorm
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle


DEFAULT_DPI = 300
DEFAULT_GRID_DIVISIONS = 16
CHANNELS = ("order", "thermal", "hpwl")
CHANNEL_LABELS = ("Priority / order", "Thermal proxy", "HPWL contribution")


def configure_colorbar(colorbar, image, label):
    """Space ticks uniformly in displayed color distance, label raw values."""
    ticks = np.asarray(image.norm.inverse(np.linspace(0., 1., 5)))
    ticks[np.abs(ticks) < 1e-14] = 0.
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([f"{v:.3g}" for v in ticks])
    colorbar.minorticks_off()
    colorbar.set_label(label, fontsize=8)
    colorbar.ax.tick_params(labelsize=8)


def compute_map_limits(maps: np.ndarray, channel: str) -> tuple[float, float]:
    """Return one scale for all supplied, already channel-sliced raw values.

    Any leading dimensions (frames, algorithms, cases) are reduced together.
    Order is normalized priority, so its scale is always [0, 1]; -1 denotes
    whitespace. Other channels include zero and use their complete raw range.
    No percentile clipping, smoothing, or per-frame normalization is applied.
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown map channel: {channel!r}")
    values = np.asarray(maps)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("map values must be nonempty and finite")
    if channel == "order":
        return 0.0, 1.0
    lo = min(0.0, float(np.min(values)))
    hi = max(0.0, float(np.max(values)))
    return lo, hi if hi > lo else lo + 1.0


def draw_map(
    axis: plt.Axes,
    values: np.ndarray,
    channel: str,
    *,
    limits: tuple[float, float] | None = None,
) -> AxesImage:
    """Display raw pixels with a channel-specific, zero-preserving color scale.

    Only whitespace is masked: -1 for order and zero for thermal/HPWL. Masking
    and color normalization never modify the input, including read-only views.
    Pass globally computed limits to compare frames, algorithms, or cases.
    """
    if channel not in CHANNELS:
        raise ValueError(f"unknown map channel: {channel!r}")
    raw = np.asarray(values)
    if raw.ndim != 2 or raw.size == 0 or not np.isfinite(raw).all():
        raise ValueError("map values must be a nonempty finite 2D array")
    lo, hi = compute_map_limits(raw, channel) if limits is None else limits
    if not np.isfinite([lo, hi]).all() or hi <= lo:
        raise ValueError("limits must be finite with vmin < vmax")

    if channel == "order":
        cmap = plt.get_cmap("viridis").resampled(16)
        norm = Normalize(vmin=lo, vmax=hi)
        mask = raw == -1
    elif channel == "thermal":
        cmap = plt.get_cmap("inferno").copy()
        norm = PowerNorm(gamma=0.6, vmin=lo, vmax=hi)
        mask = raw == 0
    else:
        cmap = plt.get_cmap("magma").copy()
        # Derive the linear region from the shared limits, never this frame.
        norm = SymLogNorm(
            linthresh=max(abs(lo), abs(hi)) * 0.01,
            linscale=0.5,
            vmin=lo,
            vmax=hi,
            base=10,
        )
        # Difference-array accumulation leaves ~1e-15 residuals outside net
        # bounding boxes. Treat only machine-precision residue as display zero;
        # retain the input and all resolvable physical contributions unchanged.
        zero_tolerance = 64 * np.finfo(np.float64).eps * max(abs(lo), abs(hi))
        mask = np.abs(raw) <= zero_tolerance
    cmap.set_bad("#f4f3ef")
    return axis.imshow(
        np.ma.array(raw, mask=mask, copy=True),
        cmap=cmap,
        norm=norm,
        origin="lower",
        interpolation="nearest",
    )


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
    if height <= 0 or width <= 0:
        raise ValueError("map dimensions must be positive")
    if divisions <= 0:
        raise ValueError("divisions must be positive")
    divisions = min(divisions, height, width)
    x_boundaries = np.linspace(-0.5, width - 0.5, divisions + 1)
    y_boundaries = np.linspace(-0.5, height - 0.5, divisions + 1)
    x_labels = np.rint(np.linspace(0, width, divisions + 1)).astype(int)
    y_labels = np.rint(np.linspace(0, height, divisions + 1)).astype(int)

    labelled = np.unique(np.rint(np.linspace(0, divisions, min(divisions, 4) + 1)).astype(int))
    unlabelled = np.setdiff1d(np.arange(divisions + 1), labelled)
    axis.set_xticks(x_boundaries[labelled])
    axis.set_yticks(y_boundaries[labelled])
    axis.set_xticklabels(x_labels[labelled], fontsize=6)
    axis.set_yticklabels(y_labels[labelled], fontsize=6)
    axis.set_xticks(x_boundaries[unlabelled], minor=True)
    axis.set_yticks(y_boundaries[unlabelled], minor=True)
    axis.tick_params(which="minor", length=0, labelbottom=False, labelleft=False)
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(-0.5, height - 0.5)
    axis.set_xlabel("raw x (px)", fontsize=7)
    axis.set_ylabel("raw y (px)", fontsize=7)
    axis.set_axisbelow(False)
    axis.grid(True, which="both", color="#7f8c99", alpha=0.28, linewidth=0.45)
    axis.set_aspect("equal", adjustable="box")


def add_chiplet_outlines(
    axis: plt.Axes,
    boxes: np.ndarray,
    *,
    height: int,
    width: int,
) -> None:
    """Overlay true normalized [x, y, width, height] bounds without labels."""
    boxes = np.asarray(boxes)
    if boxes.ndim != 2 or boxes.shape[1] != 4 or not np.isfinite(boxes).all():
        raise ValueError("chiplet_boxes must have finite shape [chiplet, 4]")
    if np.any(boxes[:, 2:] < 0):
        raise ValueError("chiplet_boxes widths and heights must be nonnegative")
    for x, y, box_width, box_height in boxes:
        axis.add_patch(
            Rectangle(
                (x * width - 0.5, y * height - 0.5),
                box_width * width,
                box_height * height,
                fill=False,
                edgecolor="#283440",
                linewidth=0.45,
                zorder=4,
            )
        )


def map_title(name: str, *, height: int, width: int, divisions: int) -> str:
    return f"{name}\nraw {width}×{height}; display grid {divisions}×{divisions}"
