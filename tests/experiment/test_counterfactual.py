from types import SimpleNamespace

import numpy as np
import pytest

from chiplet_layout_experiment.arms import Arm
from chiplet_layout_experiment.counterfactual import (
    BudgetedProblem, clone_at, finite_catalog, sample_state, summarize_labels,
)
from chiplet_layout_experiment.trajectory_problem import TracedProblem


def problem():
    return TracedProblem(
        names=("a", "b"), board_mm=np.array([10., 8.]),
        sizes=np.array([[.2, .2], [.2, .2]]), powers=np.array([4., 8.]),
        pin_indices=np.array([[0, 1]]),
        pin_offsets=np.array([[[.1, .1], [.1, .1]]]),
        initial_positions=np.array([[.1, .2], [.6, .5]]))


def identity(p, x):
    return SimpleNamespace(positions=x.copy(), valid=p._exact_overlap(x) < 1e-10,
        status="test_identity", method="test_identity", wall_seconds=0., geometry_checks=1)


def test_same_state_budget_config_deltas_and_isolation():
    p = problem()
    original = p.initial_positions.copy()
    rows = sample_state(p, original, finite_catalog(9), seeds=[3, 4], legalizer=identity)
    assert len(rows) == 12
    assert len({r["state_id"] for r in rows}) == 1
    assert len({r["input_checksum"] for r in rows}) == 1
    assert len({r["config_hash"] for r in rows}) == 6
    assert all(r["evaluator_calls"] == 9 for r in rows)
    assert all(r["postprocess_evaluator_calls"] == 1 for r in rows)
    assert all(r["gradient_calls"] == (8 if r["worker_id"] == "analytical" else 0) for r in rows)
    assert all(r["phase"] == "legal_optimization" for r in rows)
    for row in rows:
        for key in row["delta_metrics"]:
            assert row["delta_metrics"][key] == row["metrics"][key] - row["input_metrics"][key]
        assert 0 <= row["candidate_legal_rate"] <= 1
    np.testing.assert_array_equal(p.initial_positions, original)
    assert p.evaluator_calls == p.gradient_calls == 0


def test_recovery_does_not_become_wire_thermal_reward():
    p = problem()
    x = np.array([[.1, .2], [.1, .2]])
    rows = sample_state(p, x, finite_catalog(3), seeds=[3], legalizer=identity)
    assert all(r["phase"] == "feasibility_recovery" for r in rows)
    assert all(r["optimization_delta_hpwl_mm"] is None for r in rows)
    summary = summarize_labels(rows)
    assert all(g["mean_delta_hpwl_mm_legal_only"] is None for g in summary["groups"])


def test_hard_budget_cannot_be_overrun():
    p = problem()
    bounded = BudgetedProblem(p, 1)
    bounded.evaluate(p.initial_positions)
    with pytest.raises(RuntimeError, match="budget exceeded"):
        bounded.evaluate(p.initial_positions)
    assert p.evaluator_calls == 1


def test_no_implicit_start_projection_or_mixed_budgets():
    p = problem()
    x = p.initial_positions.copy()
    x[0, 0] = -.1
    with pytest.raises(ValueError, match="in bounds"):
        sample_state(p, x, finite_catalog(3), seeds=[1], legalizer=identity)
    with pytest.raises(ValueError, match="share"):
        sample_state(p, p.initial_positions,
            [Arm("a", "sa", {}, 3), Arm("b", "rl", {}, 4)], seeds=[1], legalizer=identity)


def test_clone_keeps_common_normalization_and_owns_arrays():
    p = problem()
    x = p.initial_positions + .01
    cloned = clone_at(p, x)
    assert cloned.hpwl_scale == p.hpwl_scale
    assert cloned.thermal_scale == p.thermal_scale
    cloned.sizes[0, 0] = .9
    x[:] = 0
    assert p.sizes[0, 0] == .2
    assert cloned.initial_positions[0, 0] == pytest.approx(.11)


def test_restore_legal_start_if_postprocessing_fails():
    p = problem()

    def failure(q, x):
        return SimpleNamespace(positions=np.full_like(x, .1), valid=False,
            status="failed", method="test_failure", wall_seconds=0., geometry_checks=1)

    rows = sample_state(p, p.initial_positions, finite_catalog(3), seeds=[1], legalizer=failure)
    assert all(r["rollback_to_legal_input"] for r in rows)
    assert all(r["final_legal"] for r in rows)
    assert all(r["bandit_reward"] == 0 for r in rows)


def test_report_audit_rejects_incomplete_and_duplicated_arm_grid(tmp_path):
    import json
    from chiplet_layout_experiment.counterfactual_run import write_overview
    p = problem()
    rows = sample_state(p, p.initial_positions, finite_catalog(3), seeds=[1], legalizer=identity)
    for damaged in (rows[:1], [rows[0]] * 6):
        write_overview(tmp_path, damaged, {"seeds": [1], "budget": 3})
        audit = json.loads((tmp_path / "budget_audit.json").read_text())
        assert not audit["pass"]
