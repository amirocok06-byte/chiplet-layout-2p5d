"""Same-layout, finite-configuration action labels with auditable budgets.

Extends the paired-input/checksum design of stage0-core sampling.py and the
descriptive arm grouping of stage0-baselines. It uses the continuous local
optimizers, rather than the legacy one-shot smoke workers.
"""
from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import time

import numpy as np

from .arms import Arm
from .trajectory_problem import TracedProblem
from .trajectory_optimizers import run_sa, run_analytical
from .trajectory_rl import run_policy_gradient


ALGORITHMS = {"sa": run_sa, "analytical": run_analytical, "rl": run_policy_gradient}
METRICS = ("hpwl_mm", "thermal_proxy_peak", "overlap_fraction")


def finite_catalog(budget=65):
    """Budget includes one initial evaluation and budget-1 candidate calls."""
    return (
        Arm("sa-local", "sa", {"radius": .04, "temperature_factor": .02}, budget),
        Arm("sa-explore", "sa", {"radius": .15, "temperature_factor": .1}, budget),
        Arm("analytical-small", "analytical", {"learning_rate": .01}, budget),
        Arm("analytical-large", "analytical", {"learning_rate": .06}, budget),
        Arm("rl-local", "rl", {"displacement": .02, "learning_rate": .04}, budget),
        Arm("rl-explore", "rl", {"displacement": .08, "learning_rate": .12}, budget),
    )


def array_hash(array):
    return hashlib.sha256(np.asarray(array, dtype="<f8").tobytes()).hexdigest()


def problem_hash(problem):
    digest = hashlib.sha256(json.dumps(problem.names).encode())
    for name in ("board_mm", "sizes", "powers", "pin_indices", "pin_offsets"):
        digest.update(name.encode())
        digest.update(np.asarray(getattr(problem, name), dtype="<f8").tobytes())
    digest.update(problem.VERSION.encode())
    digest.update(np.asarray([problem.hpwl_scale, problem.thermal_scale], dtype="<f8").tobytes())
    return digest.hexdigest()


def clone_at(problem, positions):
    """Own all mutable arrays; keep objective normalization fixed across forks."""
    cloned = TracedProblem(**{name: deepcopy(getattr(problem, name)) for name in (
        "names", "board_mm", "sizes", "powers", "pin_indices", "pin_offsets",
        "initial_positions")})
    cloned.initial_positions = np.array(positions, dtype=float, copy=True)
    cloned.hpwl_scale = problem.hpwl_scale
    cloned.thermal_scale = problem.thermal_scale
    return cloned


def checked_positions(problem, positions):
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (problem.n, 2) or not np.isfinite(positions).all():
        raise ValueError("state must have finite positions of shape (n,2)")
    # Optimizers project their starting point. Refuse silent changes to s_t.
    if not np.array_equal(problem.clip(positions), positions):
        raise ValueError("fork input must be in bounds, without implicit projection")
    return positions.copy()


class BudgetedProblem:
    """Enforce every objective call, including rejected candidates."""

    def __init__(self, problem, budget):
        self.problem = problem
        self.budget = budget
        self.calls = 0
        self.valid_candidates = 0
        self.evaluation_seconds = 0.

    def __getattr__(self, name):
        return getattr(self.problem, name)

    def evaluate(self, positions):
        if self.calls >= self.budget:
            raise RuntimeError("evaluation budget exceeded")
        started = time.perf_counter()
        metrics = self.problem.evaluate(positions)
        self.evaluation_seconds += time.perf_counter() - started
        if self.calls > 0:
            self.valid_candidates += int(metrics["valid"])
        self.calls += 1
        return metrics


def deltas(before, after):
    return {name: after[name] - before[name] for name in METRICS}


def _rss():
    try:
        import psutil
        return psutil.Process().memory_info().rss / 2**20
    except ImportError:
        return None


def sample_state(problem, positions, arms, *, seeds, state_metadata=None,
                 legalizer=None):
    """Each arm/seed gets the exact same state and evaluation ceiling.

    Algorithm budgets exclude the single shared baseline and post-action
    verification/legalization calls, which are separately metered. Raw and
    legalized vector deltas are both retained. No proxy is labelled Celsius.
    Legal input whose action cannot be repaired is returned unchanged; raw
    invalid action remains visible. This is an action+legalization protocol,
    and the raw protocol is available separately for ablations.
    """
    if legalizer is None:
        from .legalization import legalize
        legalizer = legalize
    arms, seeds = tuple(arms), tuple(seeds)
    if not arms or not seeds:
        raise ValueError("arms and seeds must not be empty")
    if len({a.arm_id for a in arms}) != len(arms):
        raise ValueError("arm IDs must be unique")
    if len({a.budget for a in arms}) != 1 or arms[0].budget < 2:
        raise ValueError("all arms must share an evaluation budget >= 2")
    if any(a.worker_id not in ALGORITHMS for a in arms):
        raise ValueError("unknown algorithm")
    if len(set(seeds)) != len(seeds) or any(
            isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    positions = checked_positions(problem, positions)
    context_hash = problem_hash(problem)
    position_hash = array_hash(positions)
    state_id = hashlib.sha256((context_hash + position_hash).encode()).hexdigest()
    baseline_problem = clone_at(problem, positions)
    baseline = baseline_problem.evaluate(positions)
    phase = "legal_optimization" if baseline["valid"] else "feasibility_recovery"
    rows = []
    for seed in seeds:
        # Shuffle scheduling only, not optimizer seeds; reduce systematic order bias.
        order = np.random.default_rng(seed).permutation(len(arms))
        for rank, index in enumerate(order):
            arm = arms[index]
            fork = clone_at(problem, positions)
            limited = BudgetedProblem(fork, arm.budget)
            current_valid = 0

            def callback(step, x, details):
                nonlocal current_valid
                if step:
                    current_valid += int(details["metrics"]["valid"])

            rss_before = _rss()
            started = time.perf_counter()
            result = ALGORITHMS[arm.worker_id](
                limited, steps=arm.budget - 1, seed=seed, on_step=callback,
                config=dict(arm.config))
            optimizer_seconds = time.perf_counter() - started
            if limited.calls != arm.budget:
                raise RuntimeError("optimizer did not consume the declared budget")
            raw_positions = np.asarray(result["final_positions"], dtype=float)
            raw = dict(result["final_metrics"])
            verification = clone_at(problem, positions)
            postprocess_started = time.perf_counter()
            repaired = legalizer(verification, raw_positions)
            repaired_positions = np.asarray(repaired.positions, dtype=float)
            if repaired_positions.shape != positions.shape or not np.isfinite(repaired_positions).all():
                raise ValueError("legalizer returned invalid coordinate array")
            fallback = bool(baseline["valid"] and not repaired.valid)
            output = positions.copy() if fallback else repaired_positions
            final = verification.evaluate(output)
            postprocess_seconds = time.perf_counter() - postprocess_started
            preserved = bool(baseline["valid"] and final["valid"])
            recovered = bool(not baseline["valid"] and final["valid"])
            raw_delta, final_delta = deltas(baseline, raw), deltas(baseline, final)
            # Feasibility recovery is not a wire/thermal optimization reward.
            if baseline["valid"]:
                reward = (-final_delta["hpwl_mm"] / problem.hpwl_scale
                          - .3 * final_delta["thermal_proxy_peak"] / problem.thermal_scale
                          if final["valid"] else -1.)
            else:
                reward = (float(recovered) + baseline["overlap_fraction"]
                          - final["overlap_fraction"])
            rows.append({
                "format": "same-state-action-label-v1", "state_id": state_id,
                "input_checksum": position_hash, "problem_checksum": context_hash,
                "output_checksum": array_hash(output), "phase": phase,
                "raw_output_positions": raw_positions.tolist(),
                "output_positions": output.tolist(),
                "state_metadata": deepcopy(state_metadata or {}),
                "arm_id": arm.arm_id, "worker_id": arm.worker_id,
                "config": dict(arm.config), "config_hash": arm.config_hash,
                "seed": seed, "schedule_rank": rank,
                "budget": arm.budget, "candidate_budget": arm.budget - 1,
                "evaluator_version": problem.VERSION,
                "input_metrics": dict(baseline), "raw_metrics": raw,
                "metrics": final, "raw_delta_metrics": raw_delta,
                "delta_metrics": final_delta,
                "legalization_delta_metrics": deltas(raw, final),
                "raw_final_legal": bool(raw["valid"]),
                "final_legal": bool(final["valid"]),
                "feasibility_restored": recovered,
                "legality_preserved": preserved,
                "candidate_legal_rate": limited.valid_candidates / (arm.budget - 1),
                "iterate_legal_rate": current_valid / (arm.budget - 1),
                "wire_thermal_optimization_eligible": preserved,
                "optimization_delta_hpwl_mm": final_delta["hpwl_mm"] if preserved else None,
                "optimization_delta_thermal_proxy": final_delta["thermal_proxy_peak"] if preserved else None,
                "bandit_reward": float(reward),
                "reward_definition": ("normalized wire+0.3*proxy improvement"
                    if baseline["valid"] else "restored_validity+overlap_reduction"),
                "runtime_seconds": optimizer_seconds,
                "evaluation_seconds": limited.evaluation_seconds,
                "legalization_seconds": repaired.wall_seconds,
                "postprocess_seconds": postprocess_seconds,
                "total_action_seconds": optimizer_seconds + postprocess_seconds,
                "rss_before_mb": rss_before, "rss_after_mb": _rss(),
                "memory_scope": "current process RSS, not per-arm peak or isolated allocation",
                "evaluator_calls": limited.calls,
                "gradient_calls": fork.gradient_calls,
                "baseline_evaluator_calls_shared": 1,
                "postprocess_evaluator_calls": verification.evaluator_calls,
                "legalization_geometry_checks": repaired.geometry_checks,
                "legalization_status": repaired.status,
                "legalization_method": repaired.method,
                "rollback_to_legal_input": fallback,
                "rl_initialization": "fresh zero weights per action; no pretrained claims",
                "training_steps": result.get("training_steps", 0),
            })
    if array_hash(positions) != position_hash:
        raise RuntimeError("input state was mutated")
    return rows


def summarize_labels(rows):
    """Never pool feasibility recovery with valid-to-valid metric optimization."""
    groups = defaultdict(list)
    for row in rows:
        meta = row["state_metadata"]
        groups[(meta.get("case"), meta.get("geometry_version"), row["phase"],
                row["arm_id"])].append(row)
    result = []
    for (case, version, phase, arm), entries in sorted(groups.items()):
        eligible = [r for r in entries if r["wire_thermal_optimization_eligible"]]
        mean = lambda values: float(np.mean(values)) if values else None
        result.append({
            "case": case, "geometry_version": version, "phase": phase,
            "arm_id": arm, "records": len(entries),
            "states": len({r["state_id"] for r in entries}),
            "raw_final_legal_rate": mean([r["raw_final_legal"] for r in entries]),
            "final_legal_rate": mean([r["final_legal"] for r in entries]),
            "recovery_rate": mean([r["feasibility_restored"] for r in entries])
                             if phase == "feasibility_recovery" else None,
            "mean_delta_overlap": mean([r["delta_metrics"]["overlap_fraction"] for r in entries]),
            "legal_optimization_records": len(eligible),
            "mean_delta_hpwl_mm_legal_only": mean([r["delta_metrics"]["hpwl_mm"] for r in eligible]),
            "mean_delta_thermal_proxy_legal_only": mean([r["delta_metrics"]["thermal_proxy_peak"] for r in eligible]),
            "mean_optimizer_seconds": mean([r["runtime_seconds"] for r in entries]),
            "mean_legalization_seconds": mean([r["legalization_seconds"] for r in entries]),
            "mean_candidate_legal_rate": mean([r["candidate_legal_rate"] for r in entries]),
        })
    return {"format": "same-state-stratified-summary-v1", "record_count": len(rows),
            "groups": result}
