"""NumPy-only online REINFORCE for a single persistent placement trajectory.

An action selects one chiplet and one of eight unit directions. Each bounded
proposal is evaluated once. Strict objective improvements become the next
placement; rejected proposals leave it unchanged but still train the policy
using their signed objective-improvement reward. Validity is always the
evaluator's verdict, including when the initial or final layout overlaps.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np


_DIRECTIONS = np.array([
    [1., 0.], [1., 1.], [0., 1.], [-1., 1.],
    [-1., 0.], [-1., -1.], [0., -1.], [1., -1.],
])
_DIRECTIONS /= np.linalg.norm(_DIRECTIONS, axis=1, keepdims=True)


def _features(positions: np.ndarray, sizes: np.ndarray) -> np.ndarray:
    """Bounded board coordinates, relative centres, sizes, and a bias."""
    centres = positions + sizes / 2.
    features = np.concatenate((
        (positions - .5).ravel(),
        (centres - centres.mean(axis=0)).ravel(),
        sizes.ravel(),
        [1.],
    ))
    return features / np.linalg.norm(features)


def _distribution(weights: np.ndarray, features: np.ndarray):
    logits = weights @ features
    exponentials = np.exp(logits - logits.max())
    softmax = exponentials / exponentials.sum()
    # A fixed uniform component keeps every chiplet/direction reachable.
    probabilities = .95 * softmax + .05 / len(softmax)
    return softmax, probabilities


def _weight_hash(weights: np.ndarray) -> str:
    return hashlib.sha256(weights.astype("<f8", copy=False).tobytes()).hexdigest()


def run_policy_gradient(
    problem: Any,
    *,
    steps: int,
    seed: int,
    on_step: Callable[[int, np.ndarray, dict[str, Any]], None] | None,
) -> dict[str, Any]:
    """Minimize the evaluator objective with online, baseline-centred REINFORCE.

    Policy weights, RNG, reward baseline, reward scale and per-chiplet move
    radii persist through every iteration. The learning signal is the proposal
    reward, even on rejection. A moving RMS scales advantages, and an entropy
    bonus discourages premature concentration. Both the softmax mixture's
    exact score gradient and its entropy gradient are used in each update.

    ``on_step`` receives independent position/metric copies at step zero and
    after every update. ``parameters['metrics']`` is the cached evaluation of
    the actual current placement, not the possibly rejected proposal. The
    return value separately owns current and best positions/metrics and hashes
    of the initial/final weights. Exactly ``steps + 1`` evaluations are made;
    no gradients, maps, powers or additional problem attributes are required.
    """
    if isinstance(steps, (bool, np.bool_)) or not isinstance(steps, (int, np.integer)) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    steps = int(steps)
    n = int(problem.n)
    sizes = np.array(problem.sizes, dtype=float, copy=True)
    initial = np.array(problem.initial_positions, dtype=float, copy=True)
    if n < 1 or sizes.shape != (n, 2) or initial.shape != (n, 2):
        raise ValueError("positions and sizes must have shape (n, 2), with n > 0")
    if not np.isfinite(initial).all() or not np.isfinite(sizes).all():
        raise ValueError("positions and sizes must be finite")
    if np.any(sizes <= 0) or np.any(sizes > 1):
        raise ValueError("normalized sizes must lie in (0, 1]")

    positions = np.array(problem.clip(initial), dtype=float, copy=True)
    metrics = dict(problem.evaluate(positions))
    initial_metrics = dict(metrics)
    best_positions, best_metrics = positions.copy(), dict(metrics)
    rng = np.random.default_rng(seed)
    weights = np.zeros((8 * n, len(_features(positions, sizes))))
    initial_weight_hash = _weight_hash(weights)
    displacements = np.full(n, .05)
    learning_rate, entropy_coefficient = .08, .01
    baseline = 0.
    reward_second_moment = max(.05 * abs(float(metrics["objective"])), 1e-3) ** 2
    accepted_moves, best_step = 0, 0

    def emit(step: int, details: dict[str, Any]) -> None:
        if on_step is not None:
            on_step(step, positions.copy(), {
                **details,
                "learning_rate": learning_rate,
                "entropy_coefficient": entropy_coefficient,
                "epsilon": .05,
                "baseline": float(baseline),
                "policy_weight_norm": float(np.linalg.norm(weights)),
                "actions": 8 * n,
                "training_steps": step,
                "evaluator_calls": step + 1,
                "gradient_calls": 0,
                "metrics": dict(metrics),
                "best_metrics": dict(best_metrics),
                "best_objective": float(best_metrics["objective"]),
            })

    emit(0, {
        "entropy": float(np.log(8 * n)), "reward": 0., "baseline_before": 0.,
        "advantage": 0., "update_norm": 0., "accepted": False,
        "action_index": None, "chiplet_index": None, "direction_index": None,
        "displacement": .05, "proposal_objective": None,
        "action_probability": None, "action_probability_after_update": None,
    })

    for step in range(1, steps + 1):
        features = _features(positions, sizes)
        softmax, probabilities = _distribution(weights, features)
        log_probabilities = np.log(probabilities)
        entropy = float(-probabilities @ log_probabilities)
        action = int(rng.choice(8 * n, p=probabilities))
        chiplet, direction = divmod(action, 8)
        displacement = float(displacements[chiplet])
        proposal = positions.copy()
        proposal[chiplet] += displacement * _DIRECTIONS[direction]
        proposal = np.array(problem.clip(proposal), dtype=float, copy=True)
        proposal_metrics = dict(problem.evaluate(proposal))
        reward = float(metrics["objective"] - proposal_metrics["objective"])
        baseline_before = baseline
        reward_second_moment = .95 * reward_second_moment + .05 * reward ** 2
        advantage = float(np.clip(
            (reward - baseline_before) / max(np.sqrt(reward_second_moment), 1e-8), -5., 5.,
        ))

        # d log(.95 * softmax[a] + .05/A) / d logits. The mixture
        # correction matters: using one_hot - probabilities would be biased.
        score = -softmax.copy()
        score[action] += 1.
        score *= .95 * softmax[action] / probabilities[action]
        entropy_gradient = -.95 * softmax * (
            log_probabilities - softmax @ log_probabilities
        )
        logit_gradient = advantage * score + entropy_coefficient * entropy_gradient
        logit_gradient /= max(1., float(np.linalg.norm(logit_gradient)))
        update = learning_rate * np.outer(logit_gradient, features)
        weights += update
        _, updated_probabilities = _distribution(weights, features)
        baseline = .9 * baseline_before + .1 * reward

        accepted = bool(proposal_metrics["objective"] < metrics["objective"])
        if accepted:
            positions, metrics = proposal, proposal_metrics
            accepted_moves += 1
            if metrics["objective"] < best_metrics["objective"]:
                best_positions, best_metrics = positions.copy(), dict(metrics)
                best_step = step
        displacements[chiplet] = np.clip(
            displacement * (1.03 if accepted else .97), .001, .15,
        )
        emit(step, {
            "entropy": entropy, "reward": reward, "baseline_before": float(baseline_before),
            "advantage": advantage, "update_norm": float(np.linalg.norm(update)),
            "accepted": accepted, "action_index": action, "chiplet_index": chiplet,
            "direction_index": direction, "displacement": displacement,
            "proposal_objective": float(proposal_metrics["objective"]),
            "action_probability": float(probabilities[action]),
            "action_probability_after_update": float(updated_probabilities[action]),
        })

    return {
        "method": "rl", "steps": steps, "seed": int(seed),
        "positions": positions.copy(), "metrics": dict(metrics),
        "final_positions": positions.tolist(), "final_metrics": dict(metrics),
        "initial_metrics": initial_metrics,
        "best_positions": best_positions.copy(), "best_metrics": dict(best_metrics),
        "best_step": best_step,
        "initial_weight_hash": initial_weight_hash, "final_weight_hash": _weight_hash(weights),
        "training_steps": steps, "evaluator_calls": steps + 1, "gradient_calls": 0,
        "accepted_moves": accepted_moves, "policy_weight_norm": float(np.linalg.norm(weights)),
        "state_transition": "accept_strict_objective_improvement",
        "implementation": "numpy-online-reinforce",
    }
