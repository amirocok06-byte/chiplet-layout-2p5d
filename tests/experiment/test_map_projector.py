from dataclasses import replace

import numpy as np
import pytest

from chiplet_layout_experiment.canonical import (
    Board,
    Chiplet,
    Layout,
    Net,
    PinRef,
    Placement,
)
from chiplet_layout_experiment.map_projector import rasterize_layout


def _layout() -> Layout:
    return Layout(
        board=Board(10.0, 10.0),
        chiplets=(
            Chiplet("A", 2.0, 2.0, power=4.0),
            Chiplet("B", 2.0, 2.0, power=8.0),
        ),
        placements={
            "A": Placement(1.0, 1.0),
            "B": Placement(6.0, 6.0),
        },
        nets=(
            Net(
                "N0",
                (
                    PinRef("A"),
                    PinRef("B"),
                ),
                weight=2.0,
            ),
        ),
        metadata={"chiplet_order": ("A", "B")},
    )


def test_rasterize_layout_returns_three_256_maps():
    maps = rasterize_layout(_layout(), grid_size=256)

    assert maps.shape == (3, 256, 256)
    assert maps.dtype == np.float32
    assert np.isfinite(maps).all()


def test_hpwl_map_preserves_weighted_hpwl_contribution():
    maps = rasterize_layout(_layout(), grid_size=64)

    # pin coordinates are (1, 1) and (6, 6): HPWL=10, weight=2.
    assert np.isclose(float(maps[2].sum()), 20.0, atol=1e-5)


def test_order_map_encodes_explicit_chiplet_rank():
    maps = rasterize_layout(
        _layout(),
        grid_size=64,
        order=("B", "A"),
    )

    order_map = maps[0]
    # B is rank 0 and A is rank 1; use cell centers to avoid boundaries.
    assert order_map[41, 41] == 0.0
    assert order_map[9, 9] == 1.0


def test_rasterize_layout_rejects_implicit_input_order():
    layout = replace(_layout(), metadata={})

    with pytest.raises(ValueError, match="priority/precedence order"):
        rasterize_layout(layout, grid_size=32)
