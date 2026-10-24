import gzip
import json
import numpy as np
import pytest

from chiplet_layout_experiment.trajectory_problem import TracedProblem
from chiplet_layout_experiment.trajectory_run import collect_run


@pytest.mark.parametrize("algorithm", ["sa", "analytical", "rl"])
def test_real_algorithm_recording_has_five_current_checkpoints_and_all_steps(tmp_path, algorithm):
    p = TracedProblem(
        names=("a", "b"), board_mm=np.array([10., 8.]),
        sizes=np.array([[.2, .2], [.2, .2]]), powers=np.array([4., 8.]),
        pin_indices=np.array([[0, 1]]),
        pin_offsets=np.array([[[.1, .1], [.1, .1]]]),
        initial_positions=np.array([[.1, .2], [.6, .5]]))
    summary = collect_run(p, {"case": "test"}, tmp_path,
                          algorithm=algorithm, steps=8, seed=7, revision="test")
    assert summary["checkpoint_steps"] == [0, 2, 4, 6, 8]
    assert summary["status"] == "completed-configured-budget"
    assert summary["evaluator_calls_total"] == 9
    assert summary["diagnostic_evaluator_calls"] == 0
    rows = [json.loads(line) for line in
            gzip.decompress((tmp_path / "trace.jsonl.gz").read_bytes()).splitlines()]
    assert [r["step"] for r in rows] == list(range(9))
    with np.load(tmp_path / "raw" / "positions.npz") as data:
        positions = data["positions"]
    with np.load(tmp_path / "raw" / "state_flow.npz") as data:
        assert data["maps"].shape == (5, 3, 256, 256)
    for checkpoint in summary["checkpoints"]:
        np.testing.assert_array_equal(checkpoint["positions"], positions[checkpoint["step"]])
        np.testing.assert_allclose(checkpoint["hpwl_map_sum_mm"],
                                   checkpoint["metrics"]["hpwl_mm"], rtol=1e-6)
    if algorithm == "rl":
        implementation = summary["implementation"]
        assert implementation["initial_weight_hash"] != implementation["final_weight_hash"]
