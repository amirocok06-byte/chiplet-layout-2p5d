from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from chiplet_layout_experiment.legalization import legalize
from chiplet_layout_experiment.trajectory_problem import TracedProblem


def problem(sizes):
    sizes = np.asarray(sizes, dtype=float)
    return SimpleNamespace(n=len(sizes), sizes=sizes, board_mm=np.array([10., 20.]))


def exact_valid(p, x):
    i, j = np.triu_indices(p.n, 1)
    lo = np.maximum(x[i], x[j])
    hi = np.minimum(x[i] + p.sizes[i], x[j] + p.sizes[j])
    fraction = np.maximum(hi - lo, 0).prod(axis=1).sum() / p.sizes.prod(axis=1).sum()
    return bool((x >= -1e-10).all() and (x + p.sizes <= 1+1e-10).all()
                and fraction < 1e-10)


def test_existing_valid_state_is_preserved_without_objective_calls():
    p = problem([[.5, 1.], [.5, 1.]])
    def forbidden(*args):
        pytest.fail("hidden objective evaluation")
    p.evaluate = forbidden
    x = np.array([[0., 0.], [.5, 0.]])
    result = legalize(p, x)
    assert result.valid and result.status == "already_valid"
    assert result.method == "identity" and result.geometry_checks == 1
    assert result.wall_seconds >= 0 and result.displacement == 0
    np.testing.assert_array_equal(result.positions, x)
    assert not np.shares_memory(result.positions, x)


@pytest.mark.parametrize("x", [
    [[.1, .1], [.2, .2]],  # complete containment
    [[-.1, .1], [.8, .8]],  # boundary violations
    [[.1, .1], [.5, .2]],  # partial overlap
])
def test_repairs_containment_boundary_and_overlap_without_mutating(x):
    p = problem([[.6, .4], [.2, .1]])
    x = np.array(x)
    before, sizes = x.copy(), p.sizes.copy()
    result = legalize(p, x)
    assert result.valid and exact_valid(p, result.positions)
    assert result.displacement > 0
    np.testing.assert_array_equal(x, before)
    np.testing.assert_array_equal(p.sizes, sizes)


@pytest.mark.parametrize("sizes,status", [
    ([[.8, .8], [.8, .8]], "proven_infeasible_by_area"),
    ([[1.1, .1]], "proven_infeasible_by_dimensions"),
])
def test_impossible_short_circuits(sizes, status):
    p = problem(sizes)
    result = legalize(p, np.zeros((p.n, 2)))
    assert not result.valid and result.status == status
    assert result.method == "proof" and result.geometry_checks == 1
    assert not result.attempts


def test_heuristic_failure_does_not_claim_infeasibility():
    p = problem([[.6, .6], [.6, .6]])  # area test cannot prove this
    result = legalize(p, np.zeros((2, 2)), max_attempts=2)
    assert not result.valid and result.status == "search_exhausted"
    assert len(result.attempts) == 2
    assert result.geometry_checks <= 4


def test_zero_budget_returns_invalid_original_without_claiming_infeasibility():
    p = problem([[.3, .3], [.3, .3]])
    result = legalize(p, np.zeros((2, 2)), max_attempts=0)
    assert not result.valid and result.status == "search_exhausted"
    assert not result.attempts


@pytest.mark.parametrize("count,side", [(61, .1), (64, .125), (100, .1)])
def test_large_dense_cases_and_determinism(count, side):
    p = problem(np.full((count, 2), side))
    x = np.full((count, 2), .45)
    first, second = legalize(p, x), legalize(p, x)
    assert first.valid and exact_valid(p, first.positions)
    np.testing.assert_array_equal(first.positions, second.positions)
    assert first.attempts == second.attempts
    assert len(first.attempts) <= 6


def test_matches_traced_evaluator_tolerances():
    p = TracedProblem(
        names=("a", "b"), board_mm=np.array([10., 20.]),
        sizes=np.full((2, 2), .2), powers=np.array([2., 3.]),
        pin_indices=np.array([[0, 1]]), pin_offsets=np.full((1, 2, 2), .1),
        initial_positions=np.array([[0., 0.], [.4, .4]]),
    )
    for x in (np.array([[-5e-11, 0.], [.4, .4]]),
              np.array([[0., 0.], [.2-1e-12, 0.]]),
              np.array([[-2e-10, 0.], [.4, .4]]),
              np.array([[0., 0.], [.2-1e-8, 0.]])):
        counters = (p.evaluator_calls, p.gradient_calls)
        result = legalize(p, x, max_attempts=0)
        assert (p.evaluator_calls, p.gradient_calls) == counters
        assert result.valid == p.evaluate(result.positions)["valid"]


@pytest.mark.parametrize("positions", [np.zeros((1, 2)), [[np.nan, 0], [0, 0]],
                                      [[np.inf, 0], [0, 0]]])
def test_malformed_positions_rejected(positions):
    with pytest.raises(ValueError, match="positions"):
        legalize(problem([[.2, .2], [.2, .2]]), positions)


@pytest.mark.parametrize("budget", [-1, 1.5, True])
def test_invalid_attempt_budget_rejected(budget):
    with pytest.raises(ValueError, match="max_attempts"):
        legalize(problem([[.2, .2]]), [[0, 0]], max_attempts=budget)
