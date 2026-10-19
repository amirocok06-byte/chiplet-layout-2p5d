from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .arms import Arm
from .canonical import Layout
from .evaluator import PhysicsEvaluator
from .workers import LayoutWorker


@dataclass(frozen=True, slots=True)
class SampleRecord:
    arm_id: str
    worker_id: str
    config_hash: str
    seed: int
    budget: int
    input_checksum: str
    output_checksum: str | None
    status: str
    metrics: Mapping[str, Any]
    evaluator_version: str | None
    runtime_seconds: float
    metadata: Mapping[str, Any] = field(default_factory=dict)
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "worker_id": self.worker_id,
            "config_hash": self.config_hash,
            "seed": self.seed,
            "budget": self.budget,
            "input_checksum": self.input_checksum,
            "output_checksum": self.output_checksum,
            "status": self.status,
            "metrics": dict(self.metrics),
            "evaluator_version": self.evaluator_version,
            "runtime_seconds": self.runtime_seconds,
            "metadata": dict(self.metadata),
            "failure_reason": self.failure_reason,
        }


def sample_arms(
    layout: Layout,
    arms: Iterable[Arm],
    *,
    workers: Mapping[str, LayoutWorker],
    evaluator: PhysicsEvaluator,
    seeds: Iterable[int],
) -> tuple[SampleRecord, ...]:
    seeds = tuple(seeds)
    if not seeds:
        raise ValueError("seeds must not be empty")
    input_checksum = layout.checksum()
    records: list[SampleRecord] = []
    for arm in arms:
        worker = workers.get(arm.worker_id)
        if worker is None:
            raise ValueError(f"no worker registered for arm {arm.arm_id!r}")
        for seed in seeds:
            result = worker.run(
                layout,
                evaluator=evaluator,
                seed=seed,
                budget=arm.budget,
                config=arm.config,
            )
            evaluation = result.evaluation
            metrics = {}
            evaluator_version = None
            if evaluation is not None:
                metrics = {
                    "valid": evaluation.valid,
                    "hpwl": evaluation.hpwl,
                    "peak_temperature": evaluation.peak_temperature,
                    "mean_temperature": evaluation.mean_temperature,
                    "evaluator_calls": evaluation.evaluator_calls,
                }
                evaluator_version = evaluation.evaluator_version
            records.append(
                SampleRecord(
                    arm_id=arm.arm_id,
                    worker_id=arm.worker_id,
                    config_hash=arm.config_hash,
                    seed=seed,
                    budget=arm.budget,
                    input_checksum=input_checksum,
                    output_checksum=result.candidate.checksum() if result.candidate else None,
                    status=result.status,
                    metrics=metrics,
                    evaluator_version=evaluator_version,
                    runtime_seconds=result.runtime_seconds,
                    metadata=result.metadata,
                    failure_reason=result.unavailable_reason,
                )
            )
    return tuple(records)


def write_jsonl(path: str | Path, records: Iterable[SampleRecord]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.as_dict(), ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def write_manifest(
    path: str | Path,
    layout: Layout,
    records: Iterable[SampleRecord],
    *,
    source_commit: str,
    evaluator_version: str,
) -> None:
    records = tuple(records)
    if not source_commit.strip():
        raise ValueError("source_commit must not be empty")
    if not evaluator_version.strip():
        raise ValueError("evaluator_version must not be empty")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "chiplet-layout-mvp1-v1",
        "input_checksum": layout.checksum(),
        "source_commit": source_commit,
        "evaluator_version": evaluator_version,
        "record_count": len(records),
        "arm_ids": sorted({record.arm_id for record in records}),
        "seeds": sorted({record.seed for record in records}),
    }
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
