import importlib
import json

import numpy as np
import pytest


class QuadraticProblem:
    """Small independent implementation of the TracedProblem contract."""

    def __init__(self, *, at_minimum=False, objective_scale=1.0):
        self.sizes = np.array([[0.1, 0.2], [0.2, 0.1], [0.15, 0.15]])
        self.target = np.array([[0.25, 0.3], [0.55, 0.6], [0.35, 0.65]])
        self.initial_positions = (
            self.target.copy()
            if at_minimum
            else np.array([[0.75, 0.7], [0.1, 0.15], [0.7, 0.1]])
        )
        self.n = len(self.sizes)
        self.objective_scale = objective_scale
        self.evaluated = []
        self.differentiated = []

    def objective(self, positions):
        return self.objective_scale * float(np.square(positions - self.target).sum())

    def evaluate(self, positions):
        self.evaluated.append(positions.copy())
        objective = self.objective(positions)
        return {
            "objective": np.float64(objective),
            "hpwl_mm": np.float64(objective),
            "thermal_proxy_peak": np.float64(1.0 + objective),
            "overlap_fraction": np.float64(0.125),
            "valid": np.bool_(False),
        }

    def gradient(self, positions, smoothing):
        self.differentiated.append((positions.copy(), smoothing))
        return 2 * self.objective_scale * (positions - self.target)

    def clip(self, positions):
        # Deliberately mutate this argument: the optimizer must own its buffers.
        np.clip(positions, 0, 1 - self.sizes, out=positions)
        return positions


def runner(name):
    module = importlib.import_module("chiplet_layout_experiment.trajectory_optimizers")
    return getattr(module, name)


def record_frames(frames):
    def on_step(step, positions, parameters):
        frames.append((step, positions.copy(), parameters.copy()))
    return on_step


@pytest.mark.parametrize("name", ["run_sa", "run_analytical"])
@pytest.mark.parametrize("steps", [0, 1, 35])
def test_budget_callbacks_metrics_and_serializable_summary(name, steps):
    problem = QuadraticProblem()
    original = problem.initial_positions.copy()
    original_sizes = problem.sizes.copy()
    frames = []
    summary = runner(name)(problem, steps=steps, seed=17, on_step=record_frames(frames))

    assert [frame[0] for frame in frames] == list(range(steps + 1))
    assert len(problem.evaluated) == summary["evaluator_calls"] == steps + 1
    expected_gradients = steps if name == "run_analytical" else 0
    assert len(problem.differentiated) == summary["gradient_calls"] == expected_gradients
    assert summary["steps"] == steps
    np.testing.assert_array_equal(problem.initial_positions, original)
    np.testing.assert_array_equal(problem.sizes, original_sizes)
    np.testing.assert_array_equal(frames[0][1], original)
    np.testing.assert_array_equal(summary["final_positions"], frames[-1][1])
    assert summary["final_metrics"] == frames[-1][2]["metrics"]
    assert summary["final_metrics"]["valid"] is False
    assert summary["final_metrics"]["overlap_fraction"] == 0.125
    assert summary["best_metrics"]["objective"] == pytest.approx(
        min(problem.objective(frame[1]) for frame in frames)
    )
    assert problem.objective(np.asarray(summary["best_positions"])) == pytest.approx(
        summary["best_metrics"]["objective"]
    )
    for step, positions, parameters in frames:
        assert np.isfinite(positions).all()
        assert (positions >= 0).all()
        assert (positions <= 1 - problem.sizes).all()
        assert parameters["evaluator_calls"] == step + 1
        assert parameters["gradient_calls"] == (step if expected_gradients else 0)
        assert parameters["metrics"]["objective"] == pytest.approx(problem.objective(positions))
        json.dumps(parameters, allow_nan=False)
    json.dumps(summary, allow_nan=False)


@pytest.mark.parametrize("name", ["run_sa", "run_analytical"])
def test_seed_reproduces_the_entire_trajectory(name):
    trajectories = []
    summaries = []
    for _ in range(2):
        frames = []
        summaries.append(runner(name)(QuadraticProblem(), steps=45, seed=9,
                                      on_step=record_frames(frames)))
        trajectories.append(frames)
    assert summaries[0] == summaries[1]
    for left, right in zip(*trajectories):
        assert left[0] == right[0]
        np.testing.assert_array_equal(left[1], right[1])
        assert left[2] == right[2]


@pytest.mark.parametrize("name", ["run_sa", "run_analytical"])
def test_callback_cannot_mutate_optimizer_state(name):
    normal = runner(name)(QuadraticProblem(), steps=30, seed=2, on_step=lambda *args: None)

    def corrupt_callback(step, positions, parameters):
        positions[:] = -999
        parameters["metrics"]["objective"] = -999
        parameters["metrics"]["valid"] = True

    observed = runner(name)(QuadraticProblem(), steps=30, seed=2, on_step=corrupt_callback)
    assert observed == normal


def test_sa_records_metropolis_acceptance_and_rejections_as_current_states():
    problem = QuadraticProblem(at_minimum=True)
    frames = []
    summary = runner("run_sa")(problem, steps=200, seed=4, on_step=record_frames(frames))
    accepted_uphill = []
    rejected = []
    for previous, current in zip(frames, frames[1:]):
        step, positions, parameters = current
        candidate = problem.evaluated[step]
        if parameters["accepted"]:
            np.testing.assert_array_equal(positions, candidate)
            if problem.objective(positions) > problem.objective(previous[1]):
                accepted_uphill.append(step)
        else:
            rejected.append(step)
            np.testing.assert_array_equal(positions, previous[1])
    assert accepted_uphill, "Metropolis must sometimes accept uphill proposals"
    assert rejected, "Rejected proposals must preserve the current state"
    assert summary["accepted_moves"] == sum(f[2]["accepted"] for f in frames[1:])
    assert problem.objective(frames[-1][1]) > summary["best_metrics"]["objective"]
    np.testing.assert_array_equal(summary["best_positions"], problem.initial_positions)


def test_sa_proposals_are_local_changes_to_the_current_state():
    problem = QuadraticProblem()
    frames = []
    runner("run_sa")(problem, steps=80, seed=3, on_step=record_frames(frames))
    for previous, current in zip(frames, frames[1:]):
        step, _, parameters = current
        delta = problem.evaluated[step] - previous[1]
        assert np.count_nonzero(np.any(delta != 0, axis=1)) <= 1
        assert np.abs(delta).max() <= parameters["proposal_radius"] + 1e-14
    temperatures = [f[2]["temperature"] for f in frames[1:]]
    radii = [f[2]["proposal_radius"] for f in frames[1:]]
    assert np.all(np.diff(temperatures) < 0)
    assert np.all(np.diff(radii) < 0)
    np.testing.assert_allclose(np.diff(np.log(temperatures)),
                               np.diff(np.log(temperatures))[0])


def test_sa_seed_changes_proposals():
    first, second = QuadraticProblem(), QuadraticProblem()
    runner("run_sa")(first, steps=5, seed=1, on_step=lambda *args: None)
    runner("run_sa")(second, steps=5, seed=2, on_step=lambda *args: None)
    assert not np.array_equal(first.evaluated[1], second.evaluated[1])


def test_analytical_uses_the_previous_current_state_and_converges():
    problem = QuadraticProblem()
    frames = []
    summary = runner("run_analytical")(problem, steps=800, seed=5,
                                       on_step=record_frames(frames))
    for step, (positions, smoothing) in enumerate(problem.differentiated):
        np.testing.assert_array_equal(positions, frames[step][1])
        assert smoothing == frames[step + 1][2]["smoothing"]
        np.testing.assert_array_equal(problem.evaluated[step + 1], frames[step + 1][1])
    assert summary["final_metrics"]["objective"] < summary["initial_metrics"]["objective"] * 1e-5
    learning_rates = [f[2]["learning_rate"] for f in frames[1:]]
    smoothings = [f[2]["smoothing"] for f in frames[1:]]
    assert np.all(np.diff(learning_rates) < 0)
    assert np.all(np.diff(smoothings) < 0)


def test_analytical_normalization_handles_large_finite_gradients():
    problem = QuadraticProblem(objective_scale=1e200)
    frames = []
    result = runner("run_analytical")(problem, steps=25, seed=1,
                                      on_step=record_frames(frames))
    assert result["final_metrics"]["objective"] < result["initial_metrics"]["objective"]
    for previous, current in zip(frames, frames[1:]):
        assert np.abs(current[1] - previous[1]).max() <= current[2]["learning_rate"] + 1e-14
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("name", ["run_sa", "run_analytical"])
@pytest.mark.parametrize("steps", [-1, 1.5, True])
def test_invalid_budget_is_rejected_before_evaluation(name, steps):
    problem = QuadraticProblem()
    with pytest.raises((TypeError, ValueError), match="steps"):
        runner(name)(problem, steps=steps, seed=1, on_step=lambda *args: None)
    assert not problem.evaluated


@pytest.mark.parametrize("name", ["run_sa", "run_analytical"])
def test_nonfinite_objective_is_rejected(name):
    problem = QuadraticProblem(objective_scale=float("nan"))
    with pytest.raises(ValueError, match="finite.*objective|objective.*finite"):
        runner(name)(problem, steps=1, seed=1, on_step=lambda *args: None)


def test_nonfinite_gradient_is_rejected():
    class NonfiniteGradient(QuadraticProblem):
        def gradient(self, positions, smoothing):
            return np.full_like(positions, np.inf)

    with pytest.raises(ValueError, match="finite.*gradient|gradient.*finite"):
        runner("run_analytical")(NonfiniteGradient(), steps=1, seed=1,
                                 on_step=lambda *args: None)
