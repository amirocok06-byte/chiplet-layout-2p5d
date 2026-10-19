from chiplet_layout_experiment.bookshelf import from_bookshelf


def _write_fixture(root):
    root.joinpath("demo.blocks").write_text(
        """UCLA blocks 1.0
NumNodes : 2
NumTerminals : 0
u0 2 3
u1 4 1
""",
        encoding="utf-8",
    )
    root.joinpath("demo.nets").write_text(
        """UCLA nets 1.0
NumNets : 1
NumPins : 2
NetDegree : 2 net_a
u0 O
u1 I
""",
        encoding="utf-8",
    )
    root.joinpath("demo.pl").write_text(
        """UCLA pl 1.0
u0 0 0 : N
u1 10 0 : N
""",
        encoding="utf-8",
    )
    root.joinpath("demo.power").write_text(
        """UCLA power 1.0
u0 1.5
u1 2.0
""",
        encoding="utf-8",
    )


def test_bookshelf_adapter_creates_canonical_layout_without_executing_source(tmp_path):
    _write_fixture(tmp_path)

    layout = from_bookshelf(tmp_path, board_width=20.0, board_height=20.0)

    assert layout.metadata["source_format"] == "bookshelf"
    assert [(chiplet.name, chiplet.power) for chiplet in layout.chiplets] == [
        ("u0", 1.5),
        ("u1", 2.0),
    ]
    assert layout.nets[0].pins[0].chiplet == "u0"
    assert layout.placements["u1"].x == 10.0
