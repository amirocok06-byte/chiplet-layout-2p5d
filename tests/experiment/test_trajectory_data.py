import numpy as np

from chiplet_layout_experiment.trajectory_data import load_case


def test_real_case_adapter_preserves_percentage_offsets_and_shared_initialization(tmp_path):
    case = tmp_path / "Case1"
    case.mkdir()
    (case / "Case1.blocks").write_text(
        "NumHardRectilinearBlocks : 2\n"
        "A hardrectilinear 4 (0, 0) (0, 1000) (2000, 1000) (2000, 0)\n"
        "B hardrectilinear 4 (0, 0) (0, 1000) (2000, 1000) (2000, 0)\n",
        encoding="utf-8")
    (case / "Case1.nets").write_text(
        "NumNets : 1\nNumPins : 2\nNetDegree : 2\n"
        "A B : %50 %-50\nB B : %-50 %50\n", encoding="utf-8")
    (case / "Case1.power").write_text("A 2\nB 4\n", encoding="utf-8")
    (case / "Case1.pl").write_text("A 0 0\nB 0 0\n", encoding="utf-8")
    first, meta = load_case(tmp_path, "Case1", seed=7)
    second, _ = load_case(tmp_path, "Case1", seed=7)
    np.testing.assert_array_equal(first.initial_positions, second.initial_positions)
    np.testing.assert_allclose(first.pin_offsets[0, 0], [2000 / 42000, 0])
    np.testing.assert_allclose(first.pin_offsets[0, 1], [0, 1000 / 42000])
    # Equal degrees are equal priorities, not an invented strict ordering.
    np.testing.assert_array_equal(first.priority, [1, 1])
    assert meta["num_nets"] == 1
    assert len(meta["files_sha256"]) == 4
