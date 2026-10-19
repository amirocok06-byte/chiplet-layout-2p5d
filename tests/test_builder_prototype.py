from __future__ import annotations

from pathlib import Path

import pytest

from chiplet_layout_workflow.builder_prototype import (
    BookshelfParseError,
    PowerThermalInput,
    ShapeInit,
    ShapeObjective,
    ShapeUpdate,
    check_non_overlap,
    check_package_boundary,
    compute_hpwl,
    deterministic_sequential_place,
    parse_bookshelf,
    run_baseline,
)


def _write_fixture(root: Path) -> Path:
    root.joinpath("demo.blocks").write_text(
        """UCLA blocks 1.0
NumNodes : 3
NumTerminals : 0
u0 2 3
u1 4 1
u2 1 2
""",
        encoding="utf-8",
    )
    root.joinpath("demo.nets").write_text(
        """UCLA nets 1.0
NumNets : 2
NumPins : 4
NetDegree : 2 net_a
u0 O
u1 I
NetDegree : 2 net_b
u1 O
u2 I
""",
        encoding="utf-8",
    )
    root.joinpath("demo.pl").write_text(
        """UCLA pl 1.0
u0 0 0 : N
u1 10 0 : N
u2 20 0 : N
""",
        encoding="utf-8",
    )
    root.joinpath("demo.power").write_text(
        """UCLA power 1.0
u0 1.5
u1 2.0
u2 0.5
""",
        encoding="utf-8",
    )
    return root


def test_parse_bookshelf_fixture_reads_geometry_nets_placements_and_power(tmp_path):
    design = parse_bookshelf(_write_fixture(tmp_path))

    assert [(chiplet.name, chiplet.width, chiplet.height) for chiplet in design.chiplets] == [
        ("u0", 2.0, 3.0),
        ("u1", 4.0, 1.0),
        ("u2", 1.0, 2.0),
    ]
    assert [(net.name, net.pins) for net in design.nets] == [
        ("net_a", ("u0", "u1")),
        ("net_b", ("u1", "u2")),
    ]
    assert design.placements["u1"].x == 10.0
    assert design.placements["u2"].orientation == "N"
    assert design.power == {"u0": 1.5, "u1": 2.0, "u2": 0.5}


def test_parse_hard_rectilinear_rectangle_case_with_pin_coordinates(tmp_path):
    tmp_path.joinpath("hard.blocks").write_text(
        """NumSoftRectangularBlocks : 0
NumHardRectilinearBlocks : 2
NumTerminals : 0

u0 hardrectilinear 4 (0, 0) (0, 3) (2, 3) (2, 0)
u1 hardrectilinear 4 (0, 0) (0, 1) (4, 1) (4, 0)
""",
        encoding="utf-8",
    )
    tmp_path.joinpath("hard.nets").write_text(
        """NumNets : 1
NumPins : 2

NetDegree : 2
u0 B : %0.5 %0.5
u1 B : %0.5 %0.5
""",
        encoding="utf-8",
    )
    tmp_path.joinpath("hard.pl").write_text(
        """u0 0 0
u1 2 0
""",
        encoding="utf-8",
    )

    design = parse_bookshelf(tmp_path)

    assert [(chiplet.name, chiplet.width, chiplet.height) for chiplet in design.chiplets] == [
        ("u0", 2.0, 3.0),
        ("u1", 4.0, 1.0),
    ]
    assert design.nets[0].pins == ("u0", "u1")


def test_sequential_placement_is_repeatable_and_checks_boundaries_and_overlap(tmp_path):
    design = parse_bookshelf(_write_fixture(tmp_path))
    first = deterministic_sequential_place(design.chiplets, package_width=6, package_height=6)
    second = deterministic_sequential_place(design.chiplets, package_width=6, package_height=6)

    assert first == second
    assert [(chiplet.name, chiplet.x, chiplet.y) for chiplet in first] == [
        ("u0", 0.0, 0.0),
        ("u1", 2.0, 0.0),
        ("u2", 0.0, 3.0),
    ]
    assert check_package_boundary(first, package_width=6, package_height=6).ok
    assert check_non_overlap(first).ok
    assert not check_package_boundary(first, package_width=5, package_height=6).ok
    assert not check_non_overlap(
        (first[0], first[1].with_position(x=1.0, y=0.0), first[2])
    ).ok


def test_compute_hpwl_and_rejects_malformed_or_missing_fields(tmp_path):
    design = parse_bookshelf(_write_fixture(tmp_path))
    placement = deterministic_sequential_place(design.chiplets, package_width=6, package_height=6)

    assert compute_hpwl(design.nets, placement) == pytest.approx(7.0)

    tmp_path.joinpath("bad.blocks").write_text(
        "UCLA blocks 1.0\nNumNodes : 1\nNumTerminals : 0\nmissing_height 2\n",
        encoding="utf-8",
    )
    tmp_path.joinpath("bad.nets").write_text(
        "UCLA nets 1.0\nNumNets : 0\nNumPins : 0\n",
        encoding="utf-8",
    )
    tmp_path.joinpath("bad.pl").write_text("UCLA pl 1.0\n", encoding="utf-8")
    with pytest.raises(BookshelfParseError, match=r"\.blocks.*line 4|missing height"):
        parse_bookshelf(tmp_path / "bad.blocks")


def test_future_shape_interfaces_and_external_adapters_are_metadata_only(tmp_path):
    design = parse_bookshelf(_write_fixture(tmp_path))
    thermal = PowerThermalInput(power_by_chiplet=design.power, ambient_temperature=25.0)

    result = run_baseline(
        tmp_path,
        package_width=6,
        package_height=6,
        thermal_input=thermal,
    )

    assert result["metrics"]["hpwl"] == pytest.approx(7.0)
    assert result["checks"]["package_boundary"] is True
    assert result["checks"]["non_overlap"] is True
    assert result["thermal_input"]["ambient_temperature"] == 25.0
    assert result["adapters"]
    assert all(adapter["executable"] is False for adapter in result["adapters"])
    assert ShapeInit and ShapeUpdate and ShapeObjective
