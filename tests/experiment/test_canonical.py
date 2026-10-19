import pytest

from chiplet_layout_experiment.canonical import (
    Board,
    Chiplet,
    Layout,
    Net,
    PinRef,
    Placement,
)


def _layout():
    return Layout(
        board=Board(width=20.0, height=20.0),
        chiplets=(
            Chiplet("a", width=4.0, height=2.0, power=10.0),
            Chiplet("b", width=3.0, height=3.0, power=5.0),
        ),
        placements={"a": Placement(1.0, 2.0), "b": Placement(10.0, 5.0)},
        nets=(
            Net(
                "n0",
                pins=(PinRef("a", 1.0, 0.5), PinRef("b", -1.0, 1.0)),
                weight=2.0,
            ),
        ),
    )


def test_canonical_layout_rejects_missing_or_extra_placements():
    with pytest.raises(ValueError, match="placements must match chiplet names"):
        Layout(
            board=Board(10.0, 10.0),
            chiplets=(Chiplet("a", 1.0, 1.0),),
            placements={"b": Placement(0.0, 0.0)},
            nets=(),
        )


def test_canonical_layout_checksum_is_stable_and_serializable():
    first = _layout()
    second = _layout()

    assert first.as_dict() == second.as_dict()
    assert first.checksum() == second.checksum()
    assert len(first.checksum()) == 64
