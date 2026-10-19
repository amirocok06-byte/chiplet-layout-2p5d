import pytest

from chiplet_layout_experiment.canonical import Board, Chiplet, Layout, Net, PinRef, Placement
from chiplet_layout_experiment.evaluator import PhysicsEvaluator


def _layout(*, overlap=False):
    return Layout(
        board=Board(20.0, 20.0),
        chiplets=(
            Chiplet("a", 4.0, 2.0, power=10.0),
            Chiplet("b", 3.0, 3.0, power=5.0),
        ),
        placements={
            "a": Placement(1.0, 2.0),
            "b": Placement(3.0 if overlap else 10.0, 3.0 if overlap else 5.0),
        },
        nets=(Net("n0", (PinRef("a", 1.0, 0.5), PinRef("b", -1.0, 1.0)), weight=2.0),),
    )


def test_physics_evaluator_uses_pin_offsets_and_net_weights():
    result = PhysicsEvaluator().evaluate(_layout())

    assert result.valid is True
    assert result.hpwl == pytest.approx(21.0)
    assert result.peak_temperature > result.ambient_temperature
    assert result.evaluator_version == "physics-v1"
    assert result.evaluator_calls == 3


def test_physics_evaluator_reports_overlap_without_hiding_metrics():
    result = PhysicsEvaluator().evaluate(_layout(overlap=True))

    assert result.valid is False
    assert result.geometry.overlap_violations == ("a overlaps b",)
    assert result.hpwl > 0.0
