from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .canonical import Board, Chiplet, Layout, Net, PinRef, Placement
from .evaluator import PhysicsEvaluator
from .map_projector import CHANNELS, rasterize_layout
from .workers import AnalyticalWorker, LayoutWorker, RLCompatibleWorker, SAWorker, WorkerResult


_EFFECT_LABELS = ("sa", "analytical", "rl-compatible")
_FLOW_LABELS = ("initial", "analytical", "sa", "rl-compatible")


def synthetic_smoke_layout() -> Layout:
    """Return a deterministic case for exercising the Stage 0 data contract only."""
    return Layout(
        board=Board(20.0, 16.0),
        chiplets=(
            Chiplet("cpu", 4.0, 3.0, power=12.0),
            Chiplet("gpu", 5.0, 4.0, power=20.0),
            Chiplet("hbm", 3.0, 5.0, power=8.0),
            Chiplet("io", 2.0, 2.0, power=3.0),
        ),
        placements={
            "cpu": Placement(12.0, 9.0),
            "gpu": Placement(1.0, 9.0),
            "hbm": Placement(14.0, 1.0),
            "io": Placement(7.0, 3.0),
        },
        nets=(
            Net("cpu-gpu", (PinRef("cpu", 2.0, 1.5), PinRef("gpu", 2.5, 2.0)), weight=2.0),
            Net("cpu-hbm", (PinRef("cpu", 3.5, 1.5), PinRef("hbm", 1.5, 2.5)), weight=3.0),
            Net("gpu-io", (PinRef("gpu", 2.5, 2.0), PinRef("io", 1.0, 1.0)), weight=1.0),
            Net("io-hbm", (PinRef("io", 1.0, 1.0), PinRef("hbm", 1.5, 2.5)), weight=1.0),
        ),
        metadata={"chiplet_order": ("hbm", "cpu", "gpu", "io")},
    )


def _workers() -> Mapping[str, LayoutWorker]:
    return {
        "sa": SAWorker(),
        "analytical": AnalyticalWorker(),
        "rl-compatible": RLCompatibleWorker(),
    }


def _metrics(result: WorkerResult) -> dict[str, Any] | None:
    evaluation = result.evaluation
    if evaluation is None:
        return None
    return {
        "valid": evaluation.valid,
        "hpwl": evaluation.hpwl,
        "peak_temperature": evaluation.peak_temperature,
        "mean_temperature": evaluation.mean_temperature,
        "evaluator_calls": evaluation.evaluator_calls,
    }


def _run_worker(
    worker: LayoutWorker,
    layout: Layout,
    *,
    evaluator: PhysicsEvaluator,
    seed: int,
) -> WorkerResult:
    return worker.run(
        layout,
        evaluator=evaluator,
        seed=seed,
        budget=1,
        config={"visualization_fixture": "synthetic-smoke"},
    )


def _result_trace(result: WorkerResult, input_layout: Layout) -> dict[str, Any]:
    return {
        "worker_id": result.worker_id,
        "status": result.status,
        "seed": result.seed,
        "budget": result.budget,
        "input_checksum": input_layout.checksum(),
        "output_checksum": result.candidate.checksum() if result.candidate else None,
        "metrics": _metrics(result),
        "runtime_seconds": result.runtime_seconds,
        "implementation": result.metadata.get("implementation"),
    }


def generate_synthetic_smoke_bundle(
    output_dir: str | Path,
    *,
    grid_size: int = 256,
    source_revision: str = "uncommitted-worktree",
) -> dict[str, Any]:
    """Write reproducible map artifacts for visualizing the Stage 0 protocol.

    This deliberately exercises local smoke workers.  It is not a benchmark,
    a reproduction of external algorithms, or evidence of algorithm preference.
    """
    destination = Path(output_dir)
    if not source_revision.strip():
        raise ValueError("source_revision must not be empty")
    destination.mkdir(parents=True, exist_ok=True)

    initial_layout = synthetic_smoke_layout()
    evaluator = PhysicsEvaluator()
    workers = _workers()
    before_maps = rasterize_layout(initial_layout, grid_size=grid_size)

    effect_maps: list[np.ndarray] = []
    effect_trace: dict[str, dict[str, Any]] = {}
    for index, label in enumerate(_EFFECT_LABELS):
        result = _run_worker(workers[label], initial_layout, evaluator=evaluator, seed=100 + index)
        if result.candidate is None:
            raise RuntimeError(f"synthetic worker {label!r} did not produce a candidate")
        effect_maps.append(rasterize_layout(result.candidate, grid_size=grid_size))
        effect_trace[label] = _result_trace(result, initial_layout)

    current_layout = initial_layout
    flow_maps = [before_maps]
    flow_trace: list[dict[str, Any]] = [
        {
            "label": "initial",
            "layout_checksum": initial_layout.checksum(),
            "metrics": _metrics(
                _run_worker(
                    workers["rl-compatible"],
                    initial_layout,
                    evaluator=evaluator,
                    seed=199,
                )
            ),
        }
    ]
    for index, label in enumerate(_FLOW_LABELS[1:]):
        result = _run_worker(
            workers[label],
            current_layout,
            evaluator=evaluator,
            seed=200 + index,
        )
        if result.candidate is None:
            raise RuntimeError(f"synthetic worker {label!r} did not produce a candidate")
        flow_trace.append({"label": label, **_result_trace(result, current_layout)})
        current_layout = result.candidate
        flow_maps.append(rasterize_layout(current_layout, grid_size=grid_size))

    np.savez_compressed(destination / "snapshot_maps.npz", maps=before_maps)
    np.savez_compressed(destination / "state_flow.npz", maps=np.stack(flow_maps))
    np.savez_compressed(
        destination / "algorithm_effects.npz",
        before_maps=before_maps,
        after_maps=np.stack(effect_maps),
    )
    manifest = {
        "format": "chiplet-layout-stage0-visual-smoke-v1",
        "synthetic": True,
        "purpose": "protocol-smoke-only",
        "warning": "Not a benchmark, external-algorithm reproduction, or preference result.",
        "grid_size": grid_size,
        "source_revision": source_revision,
        "map_channels": list(CHANNELS),
        "order_source": "explicit-priority-precedence-metadata",
        "input_layout_checksum": initial_layout.checksum(),
        "labels": list(_EFFECT_LABELS),
        "state_flow_labels": list(_FLOW_LABELS),
        "effect_trace": effect_trace,
        "state_flow_trace": flow_trace,
        "evaluator_version": evaluator.VERSION,
        "worker_implementations": {
            label: trace["implementation"] for label, trace in effect_trace.items()
        },
    }
    (destination / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate traceable Stage 0 synthetic-smoke map artifacts."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--grid-size", type=int, default=256)
    parser.add_argument("--source-revision", default="uncommitted-worktree")
    args = parser.parse_args()
    generate_synthetic_smoke_bundle(
        args.output_dir,
        grid_size=args.grid_size,
        source_revision=args.source_revision,
    )


if __name__ == "__main__":
    main()
