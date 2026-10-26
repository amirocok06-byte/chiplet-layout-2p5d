"""Transparent local SA and analytical baselines for continuous trajectories.

These implementations are not reproductions of TAP or ATPlace. They keep the
actual current iterate, including rejected SA moves and any remaining overlap;
best-so-far positions are reported separately and never replace recorded frames.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
import math
from numbers import Real
import operator
from typing import Any, Callable, Protocol

import numpy as np


class _Problem(Protocol):
    initial_positions: np.ndarray
    sizes: np.ndarray
    n: int

    def evaluate(self, positions: np.ndarray) -> dict: ...

    def gradient(self, positions: np.ndarray, smoothing: float) -> np.ndarray: ...

    def clip(self, positions: np.ndarray) -> np.ndarray: ...


_Callback = Callable[[int, np.ndarray, dict], Any]


def _numeric_config(config: Mapping[str, float] | None, defaults: dict[str, float],
                    *, unit_interval: tuple[str, ...] = (),
                    nonnegative: tuple[str, ...] = ()) -> dict[str, float]:
    """Copy and validate numeric overrides before any problem evaluation."""
    if config is None:
        return defaults.copy()
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping of numeric parameters")
    unknown = config.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"unknown config parameters: {', '.join(sorted(map(str, unknown)))}")
    values = defaults.copy()
    for name, value in config.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
            raise TypeError(f"{name} must be a finite real number")
        try:
            value = float(value)
        except OverflowError as exc:
            raise ValueError(f"{name} must be finite") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < 0 or (value == 0 and name not in nonnegative):
            bound = "nonnegative" if name in nonnegative else "positive"
            raise ValueError(f"{name} must be {bound}")
        if name in unit_interval and value > 1:
            raise ValueError(f"{name} must lie in (0, 1]")
        values[name] = value
    return values


def _nonnegative_integer(value: int, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a nonnegative integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be a nonnegative integer") from exc
    if value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return int(value)


def _project(problem: _Problem, positions: np.ndarray) -> np.ndarray:
    positions = np.array(problem.clip(np.array(positions, dtype=float, copy=True)),
                         dtype=float, copy=True)
    if positions.shape != (problem.n, 2) or not np.isfinite(positions).all():
        raise ValueError("positions must be a finite array of shape (n, 2)")
    return positions


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _evaluate(problem: _Problem, positions: np.ndarray) -> dict:
    metrics = _plain(problem.evaluate(positions.copy()))
    metrics["objective"] = float(metrics["objective"])
    if not math.isfinite(metrics["objective"]):
        raise ValueError("evaluation objective must be finite")
    return metrics


def _emit(on_step: _Callback, step: int, positions: np.ndarray, metrics: dict,
          gradient_calls: int, **parameters: Any) -> None:
    # Callback buffers are snapshots: consumers may retain or modify them.
    on_step(step, positions.copy(), deepcopy({
        **parameters,
        "metrics": metrics,
        "evaluator_calls": step + 1,
        "gradient_calls": gradient_calls,
    }))


def _summary(method: str, steps: int, seed: int, current: np.ndarray,
             initial_metrics: dict, current_metrics: dict, best: np.ndarray,
             best_metrics: dict, best_step: int, gradient_calls: int,
             accepted_moves: int) -> dict:
    return {
        "method": method,
        "implementation": (
            "local_metropolis_sa" if method == "sa"
            else "local_projected_gradient_descent"
        ),
        "steps": steps,
        "seed": seed,
        "evaluator_calls": steps + 1,
        "gradient_calls": gradient_calls,
        "accepted_moves": accepted_moves,
        "initial_metrics": deepcopy(initial_metrics),
        "final_metrics": deepcopy(current_metrics),
        "best_metrics": deepcopy(best_metrics),
        "best_step": best_step,
        "final_positions": current.tolist(),
        "best_positions": best.tolist(),
    }


def run_sa(problem: _Problem, *, steps: int, seed: int, on_step: _Callback,
           config: Mapping[str, float] | None = None) -> dict:
    """Run single-chiplet Metropolis SA with geometric cooling and move decay.

    There are exactly ``steps + 1`` evaluations and no gradient calls. The
    initial temperature is 5% of the initial objective scale (at least one),
    cooling by 1e-3; the normalized proposal radius shrinks from .12 to .006.
    Every callback includes current ``metrics`` and both operation counters.

    ``config`` overrides temperature_factor=.05, cooling_ratio=.001,
    radius=.12 and radius_decay=.05. Values must be finite and positive;
    ratios and the normalized radius must be at most one.
    """
    steps = _nonnegative_integer(steps, "steps")
    seed = _nonnegative_integer(seed, "seed")
    parameters = _numeric_config(config, {
        "temperature_factor": .05, "cooling_ratio": .001,
        "radius": .12, "radius_decay": .05,
    }, unit_interval=("cooling_ratio", "radius", "radius_decay"))
    rng = np.random.default_rng(seed)
    current = _project(problem, problem.initial_positions)
    initial_metrics = current_metrics = _evaluate(problem, current)
    best, best_metrics, best_step = current.copy(), current_metrics, 0
    initial_temperature = parameters["temperature_factor"] * max(1.0, abs(current_metrics["objective"]))
    accepted_moves = 0
    _emit(on_step, 0, current, current_metrics, 0,
          temperature=initial_temperature, proposal_radius=parameters["radius"],
          accepted=False, move="initial", best_objective=best_metrics["objective"])

    for step in range(1, steps + 1):
        progress = (step - 1) / max(steps - 1, 1)
        temperature = initial_temperature * parameters["cooling_ratio"] ** progress
        radius = parameters["radius"] * parameters["radius_decay"] ** progress
        candidate = current.copy()
        if problem.n:
            selected = int(rng.integers(problem.n))
            candidate[selected] += rng.uniform(-radius, radius, size=2)
        candidate = _project(problem, candidate)
        candidate_metrics = _evaluate(problem, candidate)
        objective = current_metrics["objective"]
        proposed_objective = candidate_metrics["objective"]
        # Log-space Metropolis avoids exp overflow/underflow for large costs.
        accepted = proposed_objective <= objective
        if not accepted:
            log_uniform = math.log(max(float(rng.random()), np.finfo(float).tiny))
            accepted = log_uniform < (objective - proposed_objective) / temperature
        if accepted:
            current, current_metrics = candidate, candidate_metrics
            accepted_moves += 1
            if current_metrics["objective"] < best_metrics["objective"]:
                best, best_metrics, best_step = current.copy(), current_metrics, step
        _emit(on_step, step, current, current_metrics, 0,
              temperature=temperature, proposal_radius=radius,
              accepted=bool(accepted), move="translate" if problem.n else "idle",
              best_objective=best_metrics["objective"])

    return _summary("sa", steps, seed, current, initial_metrics, current_metrics,
                    best, best_metrics, best_step, 0, accepted_moves)


def run_analytical(problem: _Problem, *, steps: int, seed: int,
                   on_step: _Callback, config: Mapping[str, float] | None = None) -> dict:
    """Run deterministic projected gradient descent with smooth continuation.

    Learning rate decays geometrically from .04 to .004; smoothing decays from
    .04 to .005. Divide the gradient by max(1, its largest absolute component)
    to bound moves without squaring large gradients or amplifying tiny ones.
    One gradient and one proposal evaluation are used per iteration, with no
    line search, hidden legalization, restarts, or best-iterate substitution.
    ``seed`` is recorded for the shared API; this baseline needs no randomness.

    ``config`` overrides learning_rate=.04, learning_rate_decay=.1,
    smoothing=.04 and smoothing_decay=.125. Values must be finite and
    positive, with decay ratios at most one.
    """
    steps = _nonnegative_integer(steps, "steps")
    seed = _nonnegative_integer(seed, "seed")
    parameters = _numeric_config(config, {
        "learning_rate": .04, "learning_rate_decay": .1,
        "smoothing": .04, "smoothing_decay": .125,
    }, unit_interval=("learning_rate_decay", "smoothing_decay"))
    current = _project(problem, problem.initial_positions)
    initial_metrics = current_metrics = _evaluate(problem, current)
    best, best_metrics, best_step = current.copy(), current_metrics, 0
    _emit(on_step, 0, current, current_metrics, 0,
          learning_rate=parameters["learning_rate"], smoothing=parameters["smoothing"],
          gradient_scale=1.0,
          accepted=False, move="initial", best_objective=best_metrics["objective"])

    for step in range(1, steps + 1):
        progress = (step - 1) / max(steps - 1, 1)
        learning_rate = parameters["learning_rate"] * parameters["learning_rate_decay"] ** progress
        smoothing = parameters["smoothing"] * parameters["smoothing_decay"] ** progress
        gradient = np.array(problem.gradient(current.copy(), smoothing=smoothing),
                            dtype=float, copy=True)
        if gradient.shape != current.shape or not np.isfinite(gradient).all():
            raise ValueError("gradient must be finite and have shape (n, 2)")
        gradient_scale = float(np.max(np.abs(gradient), initial=1.0))
        current = _project(problem, current - learning_rate * (gradient / gradient_scale))
        current_metrics = _evaluate(problem, current)
        if current_metrics["objective"] < best_metrics["objective"]:
            best, best_metrics, best_step = current.copy(), current_metrics, step
        _emit(on_step, step, current, current_metrics, step,
              learning_rate=learning_rate, smoothing=smoothing,
              gradient_scale=gradient_scale, accepted=True, move="projected_gradient",
              best_objective=best_metrics["objective"])

    return _summary("analytical", steps, seed, current, initial_metrics, current_metrics,
                    best, best_metrics, best_step, steps, steps)
