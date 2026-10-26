import json
import os
from pathlib import Path
import subprocess
import sys

from chiplet_layout_experiment.stage0_summary import summarize_jsonl, summarize_records


def test_summarize_records_groups_paired_deltas_and_cost_by_arm():
    summary = summarize_records(
        (
            {
                "arm_id": "sa-balanced",
                "worker_id": "sa",
                "status": "available",
                "delta_metrics": {"hpwl": -4.0, "peak_temperature": 1.0},
                "runtime_seconds": 0.2,
                "metadata": {"total_evaluator_calls": 6},
            },
            {
                "arm_id": "sa-balanced",
                "worker_id": "sa",
                "status": "available",
                "delta_metrics": {"hpwl": -2.0, "peak_temperature": 3.0},
                "runtime_seconds": 0.4,
                "metadata": {"total_evaluator_calls": 6},
            },
            {
                "arm_id": "rl-fixed-small",
                "worker_id": "rl-compatible",
                "status": "unavailable",
                "delta_metrics": {},
                "runtime_seconds": 0.1,
                "metadata": {"total_evaluator_calls": 3},
            },
        )
    )

    assert summary["record_count"] == 3
    assert summary["available_record_count"] == 2
    assert summary["arms"][0] == {
        "arm_id": "rl-fixed-small",
        "worker_id": "rl-compatible",
        "record_count": 1,
        "available_count": 0,
        "availability_rate": 0.0,
        "mean_delta_hpwl": None,
        "mean_delta_peak_temperature": None,
        "mean_runtime_seconds": 0.1,
        "mean_total_evaluator_calls": 3.0,
    }
    assert summary["arms"][1]["mean_delta_hpwl"] == -3.0
    assert summary["arms"][1]["mean_delta_peak_temperature"] == 2.0


def test_summarize_jsonl_reads_records_and_writes_deterministic_summary(tmp_path):
    input_path = tmp_path / "samples.jsonl"
    output_path = tmp_path / "summary.json"
    input_path.write_text(
        json.dumps(
            {
                "arm_id": "analytical-wl",
                "worker_id": "analytical",
                "status": "available",
                "delta_metrics": {"hpwl": -1.0, "peak_temperature": 0.0},
                "runtime_seconds": 0.5,
                "metadata": {"total_evaluator_calls": 6},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = summarize_jsonl(input_path, output_path)

    assert summary["record_count"] == 1
    assert json.loads(output_path.read_text(encoding="utf-8")) == summary


def test_stage0_summary_module_entrypoint_writes_json(tmp_path):
    project_root = Path(__file__).parents[2]
    environment = {**os.environ, "PYTHONPATH": str(project_root / "src")}
    input_path = tmp_path / "samples.jsonl"
    output_path = tmp_path / "summary.json"
    input_path.write_text(
        json.dumps(
            {
                "arm_id": "analytical-wl",
                "worker_id": "analytical",
                "status": "available",
                "delta_metrics": {"hpwl": -1.0, "peak_temperature": 0.0},
                "runtime_seconds": 0.5,
                "metadata": {"total_evaluator_calls": 6},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "chiplet_layout_experiment.stage0_summary",
            str(input_path),
            str(output_path),
        ],
        check=False,
        cwd=project_root,
        env=environment,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(output_path.read_text(encoding="utf-8"))["record_count"] == 1
