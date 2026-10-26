from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Iterable, Mapping
import json
from pathlib import Path
from statistics import fmean
from typing import Any


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _mean(values: Iterable[float]) -> float | None:
    collected = tuple(values)
    return fmean(collected) if collected else None


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Produce descriptive, per-arm statistics from Stage 0 JSONL records.

    The summary intentionally reports no winner or significance claim.  It is a
    protocol audit: paired deltas are only meaningful when records share the
    same case, state, evaluator, and budget as specified by the experiment plan.
    """
    collected = tuple(dict(record) for record in records)
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in collected:
        arm_id = record.get("arm_id")
        worker_id = record.get("worker_id")
        if not isinstance(arm_id, str) or not arm_id:
            raise ValueError("each record must contain a non-empty arm_id")
        if not isinstance(worker_id, str) or not worker_id:
            raise ValueError("each record must contain a non-empty worker_id")
        grouped[(arm_id, worker_id)].append(record)

    arms: list[dict[str, Any]] = []
    available_total = 0
    for (arm_id, worker_id), rows in sorted(grouped.items()):
        available = [row for row in rows if row.get("status") == "available"]
        available_total += len(available)
        hpwl_deltas = [
            number
            for row in available
            for number in (_number(dict(row.get("delta_metrics") or {}).get("hpwl")),)
            if number is not None
        ]
        thermal_deltas = [
            number
            for row in available
            for number in (
                _number(dict(row.get("delta_metrics") or {}).get("peak_temperature")),
            )
            if number is not None
        ]
        runtimes = [
            number
            for row in rows
            for number in (_number(row.get("runtime_seconds")),)
            if number is not None
        ]
        evaluator_calls = [
            number
            for row in rows
            for number in (_number(dict(row.get("metadata") or {}).get("total_evaluator_calls")),)
            if number is not None
        ]
        arms.append(
            {
                "arm_id": arm_id,
                "worker_id": worker_id,
                "record_count": len(rows),
                "available_count": len(available),
                "availability_rate": len(available) / len(rows),
                "mean_delta_hpwl": _mean(hpwl_deltas),
                "mean_delta_peak_temperature": _mean(thermal_deltas),
                "mean_runtime_seconds": _mean(runtimes),
                "mean_total_evaluator_calls": _mean(evaluator_calls),
            }
        )
    return {
        "format": "chiplet-layout-stage0-summary-v1",
        "record_count": len(collected),
        "available_record_count": available_total,
        "arms": arms,
    }


def summarize_jsonl(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    source = Path(input_path)
    rows: list[Mapping[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise ValueError(f"JSONL row {line_number} must be an object")
        rows.append(row)

    summary = summarize_records(rows)
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize paired Stage 0 JSONL records by finite arm."
    )
    parser.add_argument("input_path", type=Path)
    parser.add_argument("output_path", type=Path)
    args = parser.parse_args()
    summarize_jsonl(args.input_path, args.output_path)


if __name__ == "__main__":
    main()
