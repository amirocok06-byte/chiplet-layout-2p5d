"""Numeric configs affect actual trajectories without changing legacy defaults."""
import hashlib
import json
from types import MappingProxyType

import numpy as np
import pytest

from chiplet_layout_experiment.trajectory_optimizers import run_analytical, run_sa
from chiplet_layout_experiment.trajectory_rl import run_policy_gradient


DEFAULTS = {
    "sa": dict(temperature_factor=.05, cooling_ratio=.001, radius=.12, radius_decay=.05),
    "analytical": dict(learning_rate=.04, learning_rate_decay=.1,
                       smoothing=.04, smoothing_decay=.125),
    "rl": dict(learning_rate=.08, entropy_coefficient=.01, displacement=.05),
}
RUNNERS = {"sa": run_sa, "analytical": run_analytical, "rl": run_policy_gradient}
# Captured from the original implementations before adding config support.
LEGACY_HASHES = {
    "sa": "18e9f90d028dacf8e67831c2d21420cf6ea3740aa3a6971c2eeda58fd0351d15",
    "analytical": "34f84c1c2e2a42a9a6d7efbb8cfcfcf3f2beb07fe6c3aab92484df5f9fdba3ba",
    "rl": "7b670f057ccd636d76792fc9754257333f4ca07812b52f15941f322e3f479edf",
}


class SmoothProblem:
    """A smooth absolute-distance objective whose gradient uses smoothing."""

    def __init__(self):
        self.initial_positions = np.array([[.15, .2], [.65, .6]])
        self.sizes = np.array([[.1, .12], [.15, .1]])
        self.target = np.array([[.42, .35], [.38, .4]])
        self.n = 2
        self.evaluated = []
        self.smoothings = []

    def clip(self, positions):
        return np.clip(positions, 0., 1. - self.sizes)

    def evaluate(self, positions):
        self.evaluated.append(positions.copy())
        return {"objective": float(np.abs(positions - self.target).sum()), "valid": False}

    def gradient(self, positions, smoothing):
        self.smoothings.append(smoothing)
        delta = positions - self.target
        return delta / np.sqrt(delta ** 2 + smoothing ** 2)


def trace(method, *, steps=300, seed=17, **kwargs):
    problem, frames = SmoothProblem(), []
    result = RUNNERS[method](
        problem, steps=steps, seed=seed,
        on_step=lambda step, positions, params: frames.append((step, positions, params)),
        **kwargs,
    )
    return problem, frames, result


def digest(value):
    payload = json.dumps(value, sort_keys=True, allow_nan=False,
                         default=lambda array: array.tolist())
    return hashlib.sha256(payload.encode()).hexdigest()


@pytest.mark.parametrize("method", RUNNERS)
def test_default_trace_matches_pre_config_implementation(method):
    problem, frames, result = trace(method)
    assert digest((problem.evaluated, frames, result)) == LEGACY_HASHES[method]


@pytest.mark.parametrize("method", RUNNERS)
@pytest.mark.parametrize("steps", [0, 1, 300])
def test_none_empty_and_explicit_defaults_are_exactly_equivalent(method, steps):
    baseline = trace(method, steps=steps)
    expected = digest((baseline[0].evaluated, baseline[1:]))
    for config in (None, {}, DEFAULTS[method].copy()):
        problem, frames, result = trace(method, steps=steps, config=config)
        assert digest((problem.evaluated, (frames, result))) == expected
        assert len(problem.evaluated) == result["evaluator_calls"] == steps + 1


CHANGES = [
    ("sa", "temperature_factor", .0001), ("sa", "cooling_ratio", 1.),
    ("sa", "radius", .03), ("sa", "radius_decay", 1.),
    ("analytical", "learning_rate", .01),
    ("analytical", "learning_rate_decay", 1.),
    ("analytical", "smoothing", .3), ("analytical", "smoothing_decay", 1.),
    ("rl", "learning_rate", 2.), ("rl", "entropy_coefficient", 5.),
    ("rl", "displacement", .12),
]


@pytest.mark.parametrize("method,param,value", CHANGES)
def test_each_parameter_changes_actual_positions(method, param, value):
    _, original, _ = trace(method)
    config = {param: value}
    problem, changed, result = trace(method, config=config)
    assert config == {param: value}, "caller config must not be mutated"
    assert not np.array_equal([f[1] for f in original], [f[1] for f in changed])
    assert len(problem.evaluated) == result["evaluator_calls"] == 301
    assert len(changed) == 301


@pytest.mark.parametrize("method", RUNNERS)
@pytest.mark.parametrize("config", [{"typo": .1}, [], 0, False, "radius"])
def test_invalid_config_rejected_before_evaluation(method, config):
    problem = SmoothProblem()
    with pytest.raises((TypeError, ValueError), match="config|Unknown|unknown"):
        RUNNERS[method](problem, steps=0, seed=17, on_step=lambda *a: None, config=config)
    assert problem.evaluated == []


PARAMS = [(method, param) for method, params in DEFAULTS.items() for param in params]


@pytest.mark.parametrize("method,param", PARAMS)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"),
                                   -1., True, np.bool_(False), "0.1", None, 1j])
def test_invalid_parameter_rejected_before_evaluation(method, param, value):
    problem = SmoothProblem()
    with pytest.raises((TypeError, ValueError), match=param):
        RUNNERS[method](problem, steps=0, seed=17, on_step=lambda *a: None,
                        config={param: value})
    assert problem.evaluated == []


@pytest.mark.parametrize("method,param", [p for p in PARAMS if p[1] != "entropy_coefficient"])
def test_positive_parameters_reject_zero(method, param):
    with pytest.raises(ValueError, match=param):
        trace(method, steps=0, config={param: 0.})


@pytest.mark.parametrize("method,param", [p for p in PARAMS if p[1] in {
    "cooling_ratio", "radius", "radius_decay", "learning_rate_decay",
    "smoothing_decay", "displacement",
}])
def test_normalized_parameters_reject_above_one(method, param):
    with pytest.raises(ValueError, match=param):
        trace(method, steps=0, config={param: 1.01})


def test_zero_entropy_is_supported_and_numpy_numeric_values_are_accepted():
    _, frames, _ = trace("rl", steps=2, config={
        "entropy_coefficient": np.float64(0.), "learning_rate": np.int64(1),
    })
    assert all(f[2]["entropy_coefficient"] == 0. for f in frames)
    assert all(f[2]["learning_rate"] == 1. for f in frames)


def test_sa_custom_geometric_schedules_reach_configured_endpoints():
    config = MappingProxyType(dict(temperature_factor=.2, cooling_ratio=.25,
                                   radius=.3, radius_decay=.5))
    _, frames, _ = trace("sa", steps=3, config=config)
    scale = max(1., abs(frames[0][2]["metrics"]["objective"]))
    np.testing.assert_allclose([f[2]["temperature"] for f in frames],
                               .2 * scale * np.array([1., 1., .5, .25]))
    np.testing.assert_allclose([f[2]["proposal_radius"] for f in frames],
                               .3 * np.array([1., 1., np.sqrt(.5), .5]))


def test_analytical_custom_smoothing_reaches_gradient_and_callback():
    problem, frames, _ = trace("analytical", steps=3, config={
        "learning_rate": .2, "learning_rate_decay": .25,
        "smoothing": .3, "smoothing_decay": .5,
    })
    np.testing.assert_allclose([f[2]["learning_rate"] for f in frames], [.2, .2, .1, .05])
    np.testing.assert_allclose(problem.smoothings, .3 * np.array([1., np.sqrt(.5), .5]))
    assert problem.smoothings == [f[2]["smoothing"] for f in frames[1:]]


def test_rl_configured_displacement_controls_first_proposal():
    problem, frames, _ = trace("rl", steps=1, config={"displacement": .12})
    assert frames[0][2]["displacement"] == frames[1][2]["displacement"] == .12
    assert np.linalg.norm(problem.evaluated[1] - problem.evaluated[0]) == pytest.approx(.12)


@pytest.mark.parametrize("method,param", [p for p in PARAMS if p[1] in {
    "cooling_ratio", "radius", "radius_decay", "learning_rate_decay",
    "smoothing_decay", "displacement",
}])
def test_normalized_upper_boundary_is_supported(method, param):
    trace(method, steps=2, config={param: 1.})
