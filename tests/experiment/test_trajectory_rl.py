"""Behavioral tests for the persistent NumPy online policy optimizer."""
import json

import numpy as np
import pytest

from chiplet_layout_experiment.trajectory_rl import run_policy_gradient


class QuadraticProblem:
    def __init__(self, *, target=None, valid=True):
        self.initial_positions = np.array([[.15, .20], [.65, .60]])
        self.sizes = np.array([[.10, .12], [.15, .10]])
        self.n = len(self.sizes)
        self.target = np.full((self.n, 2), .42) if target is None else target
        self.valid = valid
        self.evaluated = []

    def clip(self, positions):
        return np.clip(positions, 0., 1. - self.sizes)

    def evaluate(self, positions):
        self.evaluated.append(positions.copy())
        objective = float(np.square(positions - self.target).sum())
        return {
            "objective": objective,
            "hpwl_mm": objective,
            "thermal_proxy_peak": 1.,
            "overlap_fraction": 0. if self.valid else .1,
            "valid": self.valid,
        }


def run_trace(problem, *, steps=80, seed=17):
    trace = []

    def record(step, positions, parameters):
        trace.append((step, positions.copy(), parameters))

    return run_policy_gradient(problem, steps=steps, seed=seed, on_step=record), trace


def test_quadratic_improves_with_real_updates_and_exact_evaluation_budget():
    problem = QuadraticProblem()
    original = problem.initial_positions.copy()
    result, trace = run_trace(problem, steps=800)

    assert len(problem.evaluated) == result["evaluator_calls"] == 801
    assert result["training_steps"] == 800
    assert result["metrics"]["objective"] < .1 * trace[0][2]["metrics"]["objective"]
    assert result["initial_weight_hash"] != result["final_weight_hash"]
    assert len(result["initial_weight_hash"]) == len(result["final_weight_hash"]) == 64
    assert any(p["update_norm"] > 0 for _, _, p in trace[1:])
    assert trace[-1][2]["policy_weight_norm"] > 0
    np.testing.assert_array_equal(problem.initial_positions, original)
    np.testing.assert_array_equal(result["positions"], trace[-1][1])
    assert result["metrics"] == trace[-1][2]["metrics"]
    assert result["best_metrics"]["objective"] == min(
        p["metrics"]["objective"] for _, _, p in trace
    )
    assert not np.shares_memory(result["positions"], result["best_positions"])


def test_seed_reproduces_training_and_prefix_is_one_persistent_run():
    first, first_trace = run_trace(QuadraticProblem(), steps=60, seed=7)
    second, second_trace = run_trace(QuadraticProblem(), steps=60, seed=7)
    _, short_trace = run_trace(QuadraticProblem(), steps=20, seed=7)

    assert first["final_weight_hash"] == second["final_weight_hash"]
    for a, b in zip(first_trace, second_trace):
        assert a[0] == b[0]
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2]
    for a, b in zip(first_trace, short_trace):
        np.testing.assert_array_equal(a[1], b[1])
        assert a[2] == b[2]
    other, _ = run_trace(QuadraticProblem(), steps=60, seed=8)
    assert first["final_weight_hash"] != other["final_weight_hash"]


def test_callbacks_allow_five_snapshots_without_hidden_evaluations():
    problem = QuadraticProblem()
    result, trace = run_trace(problem, steps=8)
    assert [step for step, _, _ in trace] == list(range(9))
    assert len([entry for entry in trace if entry[0] in {0, 2, 4, 6, 8}]) == 5
    for step, positions, parameters in trace:
        assert parameters["actions"] == 8 * problem.n
        assert parameters["training_steps"] == step
        assert parameters["evaluator_calls"] == step + 1
        assert parameters["learning_rate"] > 0
        assert np.isfinite(parameters["entropy"])
        assert {"reward", "baseline", "update_norm", "policy_weight_norm", "accepted"} <= parameters.keys()
        assert parameters["metrics"]["objective"] == pytest.approx(
            float(np.square(positions - problem.target).sum())
        )
        json.dumps(parameters, allow_nan=False)
    assert len(problem.evaluated) == result["evaluator_calls"] == 9


def test_rejected_proposals_train_from_their_negative_reward():
    problem = QuadraticProblem()
    problem.target = problem.initial_positions.copy()
    result, trace = run_trace(problem, steps=1)
    parameters = trace[1][2]
    assert parameters["reward"] < 0
    assert parameters["accepted"] is False
    assert parameters["update_norm"] > 0
    assert parameters["action_probability_after_update"] < parameters["action_probability"]
    assert result["initial_weight_hash"] != result["final_weight_hash"]
    np.testing.assert_array_equal(trace[0][1], trace[1][1])
    assert parameters["metrics"] == trace[0][2]["metrics"]
    assert parameters["proposal_objective"] > parameters["metrics"]["objective"]


def test_reward_sign_changes_selected_action_probability():
    # Discover the seeded first proposal, then make that exact proposal optimal.
    probe = QuadraticProblem()
    run_trace(probe, steps=1)
    proposal = probe.evaluated[1]
    rewarded = QuadraticProblem(target=proposal)
    _, trace = run_trace(rewarded, steps=1)
    parameters = trace[1][2]
    assert parameters["reward"] > 0
    assert parameters["accepted"] is True
    assert parameters["action_probability_after_update"] > parameters["action_probability"]


def test_transition_uses_current_state_and_adaptive_single_chiplet_moves():
    problem = QuadraticProblem()
    _, trace = run_trace(problem, steps=100)
    assert any(p["accepted"] for _, _, p in trace[1:])
    assert any(not p["accepted"] for _, _, p in trace[1:])
    assert len({p["displacement"] for _, _, p in trace[1:]}) > 1
    for previous, current, proposal in zip(trace, trace[1:], problem.evaluated[1:]):
        _, positions, parameters = current
        previous_positions = previous[1]
        moved = np.flatnonzero(np.any(proposal != previous_positions, axis=1))
        assert len(moved) <= 1
        if len(moved):
            assert moved[0] == parameters["chiplet_index"]
        assert 0 <= parameters["action_index"] < 8 * problem.n
        assert 0 <= parameters["direction_index"] < 8
        assert parameters["reward"] == pytest.approx(
            previous[2]["metrics"]["objective"] - float(np.square(proposal - problem.target).sum())
        )
        np.testing.assert_array_equal(positions, proposal if parameters["accepted"] else previous_positions)
        assert (positions >= 0).all()
        assert (positions + problem.sizes <= 1).all()
        assert parameters["baseline_before"] == previous[2]["baseline"]
        assert parameters["baseline"] == pytest.approx(
            .9 * parameters["baseline_before"] + .1 * parameters["reward"]
        )


def test_invalid_evaluator_result_is_not_reported_as_valid():
    result, trace = run_trace(QuadraticProblem(valid=False), steps=10)
    assert result["metrics"]["valid"] is False
    assert result["best_metrics"]["valid"] is False
    assert all(p["metrics"]["valid"] is False for _, _, p in trace)


def test_callback_mutation_cannot_corrupt_optimizer_or_older_snapshots():
    expected, trace = run_trace(QuadraticProblem(), steps=30)

    def mutate(step, positions, parameters):
        positions[:] = 900
        parameters["metrics"]["objective"] = -900

    actual = run_policy_gradient(QuadraticProblem(), steps=30, seed=17, on_step=mutate)
    assert actual["final_weight_hash"] == expected["final_weight_hash"]
    np.testing.assert_array_equal(actual["positions"], expected["positions"])
    assert actual["metrics"] == expected["metrics"]
    assert trace[0][2]["metrics"]["objective"] > trace[-1][2]["metrics"]["objective"]


def test_zero_steps_emits_initial_snapshot_without_training():
    result, trace = run_trace(QuadraticProblem(), steps=0)
    assert len(trace) == result["evaluator_calls"] == 1
    assert result["training_steps"] == 0
    assert result["initial_weight_hash"] == result["final_weight_hash"]
    assert trace[0][2]["action_index"] is None
    assert trace[0][2]["accepted"] is False
    assert trace[0][2]["update_norm"] == 0


def test_traced_problem_integration_preserves_evaluator_metrics_and_counters():
    from chiplet_layout_experiment.trajectory_problem import TracedProblem

    problem = TracedProblem(
        names=("a", "b"), board_mm=np.array([10., 8.]),
        sizes=np.array([[.4, .4], [.4, .4]]), powers=np.array([4., 8.]),
        pin_indices=np.array([[0, 1]]),
        pin_offsets=np.array([[[.2, .2], [.2, .2]]]),
        initial_positions=np.array([[.1, .2], [.2, .3]]),
    )
    result, trace = run_trace(problem, steps=800)
    assert result["evaluator_calls"] == problem.evaluator_calls == 801
    assert result["gradient_calls"] == problem.gradient_calls == 0
    assert len([step for step, _, _ in trace if step % 200 == 0]) == 5
    assert trace[0][2]["metrics"]["valid"] is False
    assert result["final_metrics"] == result["metrics"]
    assert result["metrics"] == problem.evaluate(result["positions"])
    assert result["metrics"]["valid"] == (result["metrics"]["overlap_fraction"] < 1e-10)


@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_step_budget_fails_before_evaluation(steps):
    problem = QuadraticProblem()
    with pytest.raises(ValueError, match="steps"):
        run_trace(problem, steps=steps)
    assert problem.evaluated == []
