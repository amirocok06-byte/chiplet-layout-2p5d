import json

import numpy as np
import pytest

from chiplet_layout_experiment.stage1_data import load_stage1_dataset, split_by_case


def _write_case(root, case, *, incomplete=False):
    directory = root / case
    (directory / "states").mkdir(parents=True)
    problem = {
        "names": ["a", "b"],
        "board_mm": [10.0, 8.0],
        "sizes": [[0.2, 0.2], [0.2, 0.2]],
        "powers": [4.0, 8.0],
        "pin_indices": [[0, 1]],
        "pin_offsets": [[[0.1, 0.1], [0.1, 0.1]]],
        "hpwl_scale": 8.0,
        "thermal_scale": 3.0,
    }
    (directory / "problem.json").write_text(json.dumps(problem), encoding="utf-8")
    state_file = "states/state-a.json"
    (directory / state_file).write_text(json.dumps({
        "positions": [[0.1, 0.2], [0.6, 0.5]], "checksum": "position-a",
    }), encoding="utf-8")
    rows = []
    for arm, rewards in (("analytical-small", [1.0, 3.0]),
                         ("sa-local", [2.0, 4.0])):
        for seed, reward in zip((11, 12), rewards):
            if incomplete and arm == "sa-local" and seed == 12:
                continue
            rows.append({
                "state_id": "state-a", "phase": "legal_optimization",
                "arm_id": arm, "seed": seed, "budget": 65,
                "candidate_budget": 64, "evaluator_calls": 65,
                "state_metadata": {
                    "case": case, "case_split_group": case,
                    "state_file": state_file,
                },
                "input_metrics": {
                    "hpwl_mm": 16.0, "thermal_proxy_peak": 6.0,
                    "overlap_fraction": 0.0, "valid": True,
                },
                "bandit_reward": reward,
                "total_action_seconds": reward / 10,
            })
    (directory / "labels.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_load_stage1_dataset_reconstructs_fourmap_and_aggregates_seed_rewards(tmp_path):
    _write_case(tmp_path, "Case1")

    dataset = load_stage1_dataset(tmp_path, grid_size=16)

    assert dataset.arm_ids == ("analytical-small", "sa-local")
    assert len(dataset.states) == 1
    state = dataset.states[0]
    assert state.case_group == "Case1"
    assert state.phase == "legal_optimization"
    assert state.maps.shape == (4, 16, 16)
    np.testing.assert_allclose(state.rewards, [2.0, 3.0])
    np.testing.assert_allclose(state.costs, [0.2, 0.3])
    np.testing.assert_allclose(state.scalars, [2.0, 2.0, 0.0, 1.0])
    assert state.budget == 65


def test_load_stage1_dataset_rejects_incomplete_arm_seed_grid(tmp_path):
    _write_case(tmp_path, "Case1", incomplete=True)

    with pytest.raises(ValueError, match="incomplete arm/seed grid"):
        load_stage1_dataset(tmp_path, grid_size=16)


def test_split_by_case_keeps_every_state_of_one_case_in_one_partition(tmp_path):
    for case in ("Case1", "Case2", "Case4"):
        _write_case(tmp_path, case)
    dataset = load_stage1_dataset(tmp_path, grid_size=16)

    train, validation, test = split_by_case(
        dataset, train_cases=["Case1"], validation_cases=["Case2"], test_cases=["Case4"],
    )

    assert [state.case_group for state in train.states] == ["Case1"]
    assert [state.case_group for state in validation.states] == ["Case2"]
    assert [state.case_group for state in test.states] == ["Case4"]


def test_split_by_case_rejects_overlapping_or_missing_case_groups(tmp_path):
    _write_case(tmp_path, "Case1")
    _write_case(tmp_path, "Case2")
    dataset = load_stage1_dataset(tmp_path, grid_size=16)

    with pytest.raises(ValueError, match="overlap"):
        split_by_case(dataset, train_cases=["Case1"], validation_cases=["Case1"], test_cases=["Case2"])
    with pytest.raises(ValueError, match="cover"):
        split_by_case(dataset, train_cases=["Case1"], validation_cases=[], test_cases=[])
