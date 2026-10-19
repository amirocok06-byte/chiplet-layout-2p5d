from __future__ import annotations

from pathlib import Path

from chiplet_layout_workflow.builder_prototype import parse_bookshelf

from .canonical import Board, Chiplet, Layout, Net, PinRef, Placement


def from_bookshelf(
    source: str | Path,
    *,
    board_width: float,
    board_height: float,
) -> Layout:
    """Adapt a read-only Bookshelf case into the experiment IR.

    The adapter reads only the case files through the existing strict parser. It
    does not import, execute, or install any code from the source directory.
    Bookshelf pin directions are not physical offsets; they are intentionally
    represented as zero offsets until a format-specific pin adapter is added.
    """

    design = parse_bookshelf(source)
    chiplets = tuple(
        Chiplet(
            chiplet.name,
            chiplet.width,
            chiplet.height,
            power=design.power.get(chiplet.name, 0.0),
        )
        for chiplet in design.chiplets
    )
    placements = {
        name: Placement(placement.x, placement.y)
        for name, placement in design.placements.items()
    }
    nets = tuple(
        Net(net.name, tuple(PinRef(pin) for pin in net.pins), weight=1.0)
        for net in design.nets
    )
    return Layout(
        board=Board(board_width, board_height),
        chiplets=chiplets,
        placements=placements,
        nets=nets,
        metadata={"source_format": "bookshelf"},
    )
