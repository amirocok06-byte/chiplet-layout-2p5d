from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from .canonical import Layout


CHANNELS = ("order", "thermal", "hpwl")


def _grid_bounds(
    x: float,
    y: float,
    width: float,
    height: float,
    *,
    board_width: float,
    board_height: float,
    grid_size: int,
) -> tuple[int, int, int, int]:
    x0 = int(np.floor(x / board_width * grid_size))
    x1 = int(np.ceil((x + width) / board_width * grid_size))
    y0 = int(np.floor(y / board_height * grid_size))
    y1 = int(np.ceil((y + height) / board_height * grid_size))
    x0 = max(0, min(grid_size - 1, x0))
    y0 = max(0, min(grid_size - 1, y0))
    x1 = max(x0 + 1, min(grid_size, x1))
    y1 = max(y0 + 1, min(grid_size, y1))
    return x0, x1, y0, y1


def _pin_xy(layout: Layout, pin: Any) -> tuple[float, float]:
    placement = layout.placements[pin.chiplet]
    return placement.x + pin.offset_x, placement.y + pin.offset_y


def _net_bounds(
    layout: Layout,
    net: Any,
    *,
    grid_size: int,
) -> tuple[int, int, int, int, float]:
    points = [_pin_xy(layout, pin) for pin in net.pins]
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x0, x1, y0, y1 = _grid_bounds(
        min_x,
        min_y,
        max_x - min_x,
        max_y - min_y,
        board_width=layout.board.width,
        board_height=layout.board.height,
        grid_size=grid_size,
    )
    hpwl = (max_x - min_x) + (max_y - min_y)
    return x0, x1, y0, y1, hpwl


def _resolve_order(
    layout: Layout,
    order: Sequence[str] | None,
) -> tuple[str, ...]:
    if order is None:
        metadata_order = layout.metadata.get("chiplet_order")
        if not metadata_order:
            raise ValueError(
                "priority/precedence order must be provided explicitly or in layout metadata"
            )
        order = metadata_order

    resolved = tuple(order)
    names = {chiplet.name for chiplet in layout.chiplets}
    if set(resolved) != names or len(resolved) != len(names):
        raise ValueError("order must contain every chiplet exactly once")
    return resolved


def rasterize_layout(
    layout: Layout,
    *,
    grid_size: int = 256,
    order: Sequence[str] | None = None,
    thermal_field: np.ndarray | None = None,
) -> np.ndarray:
    """Return [order, thermal, hpwl] maps in float32.

    HPWL is a net-contribution field, not a copy of scalar HPWL. Each net
    contributes its weighted HPWL uniformly over its pin bounding box, so the
    map sum equals weighted HPWL up to floating-point error.

    The thermal channel is a power-density proxy unless a temperature field is
    supplied by a higher-fidelity evaluator.
    """
    if isinstance(grid_size, bool) or not isinstance(grid_size, int) or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer")

    maps = np.zeros((3, grid_size, grid_size), dtype=np.float32)
    # Rank zero is a valid priority value.  Keep it distinct from whitespace so
    # the lowest-priority chiplet is visible to both plots and learned models.
    maps[0].fill(-1.0)
    resolved_order = _resolve_order(layout, order)
    rank_denominator = max(len(resolved_order) - 1, 1)
    rank_by_name = {
        name: index / rank_denominator
        for index, name in enumerate(resolved_order)
    }

    for chiplet in layout.chiplets:
        placement = layout.placements[chiplet.name]
        x0, x1, y0, y1 = _grid_bounds(
            placement.x,
            placement.y,
            chiplet.width,
            chiplet.height,
            board_width=layout.board.width,
            board_height=layout.board.height,
            grid_size=grid_size,
        )
        maps[0, y0:y1, x0:x1] = rank_by_name[chiplet.name]
        density = chiplet.power / (chiplet.width * chiplet.height)
        maps[1, y0:y1, x0:x1] = density

    if thermal_field is not None:
        thermal = np.asarray(thermal_field, dtype=np.float32)
        if thermal.shape != (grid_size, grid_size):
            raise ValueError("thermal_field must match (grid_size, grid_size)")
        if not np.isfinite(thermal).all():
            raise ValueError("thermal_field contains NaN or Inf")
        maps[1] = thermal

    for net in layout.nets:
        x0, x1, y0, y1, hpwl = _net_bounds(
            layout,
            net,
            grid_size=grid_size,
        )
        cell_count = max((x1 - x0) * (y1 - y0), 1)
        maps[2, y0:y1, x0:x1] += net.weight * hpwl / cell_count

    return maps


def order_transition_map(
    layout: Layout,
    *,
    grid_size: int = 256,
    order: Sequence[str] | None = None,
) -> np.ndarray:
    """Rasterize consecutive order relations as a diagnostic map."""
    resolved_order = _resolve_order(layout, order)
    result = np.zeros((grid_size, grid_size), dtype=np.float32)
    centers: dict[str, tuple[float, float]] = {}

    for chiplet in layout.chiplets:
        placement = layout.placements[chiplet.name]
        centers[chiplet.name] = (
            placement.x + chiplet.width / 2,
            placement.y + chiplet.height / 2,
        )

    for left_name, right_name in zip(resolved_order, resolved_order[1:]):
        x0, y0 = centers[left_name]
        x1, y1 = centers[right_name]
        distance = max(abs(x1 - x0), abs(y1 - y0), 1.0)
        samples = max(
            int(distance / min(layout.board.width, layout.board.height) * grid_size),
            2,
        )
        for fraction in np.linspace(0.0, 1.0, samples):
            x = x0 + fraction * (x1 - x0)
            y = y0 + fraction * (y1 - y0)
            col = int(np.clip(x / layout.board.width * grid_size, 0, grid_size - 1))
            row = int(np.clip(y / layout.board.height * grid_size, 0, grid_size - 1))
            result[row, col] += 1.0

    maximum = float(result.max())
    if maximum > 0:
        result /= maximum
    return result
