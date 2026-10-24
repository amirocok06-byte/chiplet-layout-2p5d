"""Collect five genuine checkpoints and the complete iteration/parameter trace."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import numpy as np

from .trajectory_data import load_case
from .trajectory_optimizers import run_sa, run_analytical
from .trajectory_rl import run_policy_gradient


ALGORITHMS = {"sa": run_sa, "analytical": run_analytical, "rl": run_policy_gradient}


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                               default=json_value, allow_nan=False) + "\n", encoding="utf-8")


def collect_run(problem, source, output, *, algorithm, steps, seed, revision):
    if steps < 4 or steps % 4:
        raise ValueError("steps must be a multiple of four and at least four")
    output.mkdir(parents=True, exist_ok=True)
    raw = output / "raw"
    raw.mkdir(exist_ok=True)
    checkpoint_ids = [0, steps // 4, steps // 2, 3 * steps // 4, steps]
    checkpoints, traces, position_history, map_history = [], [], [], []
    diagnostic_calls, capture_seconds = 0, 0.
    initial_calls = problem.evaluator_calls
    started = time.perf_counter()

    def on_step(step, positions, parameters):
        nonlocal diagnostic_calls, capture_seconds
        capture_start = time.perf_counter()
        positions = np.array(positions, copy=True)
        parameters = dict(parameters)
        metrics = parameters.pop("metrics", None)
        if metrics is None:
            metrics = problem.evaluate(positions)
            diagnostic_calls += 1
        row = {"step": step, "metrics": metrics, "parameters": parameters,
               "position_sha256": hashlib.sha256(positions.tobytes()).hexdigest()}
        traces.append(row)
        position_history.append(positions)
        if step in checkpoint_ids:
            maps = problem.maps(positions)
            checkpoint = {**row, "positions": positions.tolist(),
                          "map_shape": list(maps.shape),
                          "hpwl_map_sum_mm": float(maps[2].sum(dtype=float)),
                          "thermal_field_max": float(maps[1].max())}
            checkpoints.append(checkpoint)
            map_history.append(maps)
            # Flush every checkpoint while the same optimizer retains its state.
            write_json(output / f"checkpoint_{step:04d}.json", checkpoint)
            np.savez_compressed(raw / f"checkpoint_{step:04d}.npz", maps=maps)
        capture_seconds += time.perf_counter() - capture_start

    algorithm_summary = ALGORITHMS[algorithm](
        problem, steps=steps, seed=seed, on_step=on_step)
    elapsed = time.perf_counter() - started
    if [r["step"] for r in traces] != list(range(steps + 1)):
        raise RuntimeError("optimizer did not emit every iteration, in order")
    if [c["step"] for c in checkpoints] != checkpoint_ids:
        raise RuntimeError("missing checkpoint")
    np.savez_compressed(raw / "state_flow.npz", maps=np.stack(map_history))
    np.savez_compressed(raw / "positions.npz", positions=np.stack(position_history))
    trace_bytes = ("\n".join(json.dumps(r, default=json_value, allow_nan=False)
                             for r in traces) + "\n").encode()
    (output / "trace.jsonl.gz").write_bytes(gzip.compress(trace_bytes, mtime=0))
    try:
        import psutil
        memory = psutil.Process().memory_info()
        rss_mb = getattr(memory, "peak_wset", memory.rss) / 2 ** 20
    except ImportError:
        rss_mb = None
    summary = {
        "format": "real-case-local-algorithm-trajectory-v1",
        "algorithm": algorithm, "implementation": algorithm_summary,
        "source": source, "source_revision": revision,
        "algorithm_source_files": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (Path(__file__), Path(__file__).with_name("trajectory_problem.py"),
                      Path(__file__).with_name("trajectory_data.py"),
                      Path(__file__).with_name(
                          "trajectory_rl.py" if algorithm == "rl"
                          else "trajectory_optimizers.py"))},
        "status": "completed-configured-budget", "termination": "proposal-budget-exhausted",
        "steps": steps, "seed": seed, "checkpoint_steps": checkpoint_ids,
        "checkpoints": checkpoints,
        "evaluator_version": problem.VERSION,
        "evaluator_calls_total": problem.evaluator_calls - initial_calls,
        "diagnostic_evaluator_calls": diagnostic_calls,
        "gradient_calls": problem.gradient_calls,
        "wall_seconds_including_capture": elapsed,
        "capture_seconds": capture_seconds,
        "optimizer_wall_seconds": elapsed - capture_seconds,
        "peak_process_rss_mb": rss_mb,
        "python": sys.version, "numpy": np.__version__, "platform": platform.platform(),
        "objective": "HPWL/initial_HPWL + 0.3*thermal_proxy/initial_proxy + 20*penetration_penalty",
        "overlap_reporting": "exact pair intersection area / total chiplet area; distinct from penalty",
        "priority": "weighted incident-pin count / maximum; ties preserved",
        "thermal": "uncalibrated Gaussian spreading proxy; no Celsius/K claims",
        "checkpoint_display": "current iterate, not substituted best-so-far",
        "external_reproduction": False,
        "thread_limits": {n: os.environ.get(n) for n in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
    }
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", nargs="+", default=[f"Case{i}" for i in range(1, 11)])
    parser.add_argument("--algorithm", choices=ALGORITHMS)
    parser.add_argument("--steps", type=int, default=800)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--revision", default="working-tree")
    args = parser.parse_args()
    if args.algorithm:
        if len(args.cases) != 1:
            parser.error("single-algorithm subprocess requires one case")
        problem, source = load_case(args.data_root, args.cases[0], seed=args.seed)
        summary = collect_run(problem, source, args.output,
                              algorithm=args.algorithm, steps=args.steps,
                              seed=args.seed, revision=args.revision)
        print(json.dumps({"case": args.cases[0], "algorithm": args.algorithm,
                          "status": summary["status"],
                          "final": summary["checkpoints"][-1]["metrics"]}), flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
                   "OPENBLAS_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"}
    runs = []
    # Sequential, isolated subprocesses prevent competing algorithms from
    # sharing policy/RNG state or overlapping compute measurements.
    for case in args.cases:
        for algorithm in ALGORITHMS:
            destination = args.output / case / algorithm
            command = [
                sys.executable, "-m", "chiplet_layout_experiment.trajectory_run",
                "--data-root", str(args.data_root), "--output", str(destination),
                "--cases", case, "--algorithm", algorithm, "--steps", str(args.steps),
                "--seed", str(args.seed), "--revision", args.revision,
            ]
            subprocess.run(command, check=True, env=environment)
            runs.append(str((destination / "summary.json").relative_to(args.output)))
            write_json(args.output / "manifest.json",
                       {"cases": args.cases, "steps": args.steps, "seed": args.seed,
                        "runs": runs, "expected_runs": len(args.cases) * 3,
                        "status": "complete" if len(runs) == len(args.cases) * 3
                                  else "running"})


if __name__ == "__main__":
    main()
