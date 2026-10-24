import json

from chiplet_layout_experiment.arms import default_arm_catalog
from chiplet_layout_experiment.canonical import Board, Chiplet, Layout, Net, PinRef, Placement
from chiplet_layout_experiment.evaluator import PhysicsEvaluator
from chiplet_layout_experiment.sampling import sample_arms, write_jsonl, write_manifest
from chiplet_layout_experiment.workers import (
    AnalyticalWorker,
    RLCompatibleWorker,
    SAWorker,
)


def _layout():
    return Layout(
        board=Board(20.0, 20.0),
        chiplets=(Chiplet("a", 2.0, 2.0, power=1.0), Chiplet("b", 2.0, 2.0, power=2.0)),
        placements={"a": Placement(0.0, 0.0), "b": Placement(5.0, 0.0)},
        nets=(Net("n0", (PinRef("a"), PinRef("b"))),),
    )


def test_sample_arms_records_common_budget_and_auditable_metrics(tmp_path):
    workers = {
        "sa": SAWorker(),
        "analytical": AnalyticalWorker(),
        "rl-compatible": RLCompatibleWorker(),
    }
    records = sample_arms(
        _layout(),
        default_arm_catalog(),
        workers=workers,
        evaluator=PhysicsEvaluator(),
        seeds=(3,),
    )

    assert len(records) == 6
    assert {record.seed for record in records} == {3}
    assert all(record.input_checksum == _layout().checksum() for record in records)
    assert all(record.status == "available" for record in records)
    assert all("hpwl" in record.metrics for record in records)
    assert all(record.evaluator_version == "physics-v1" for record in records)

    output = tmp_path / "samples.jsonl"
    write_jsonl(output, records)
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 6
    assert rows[0]["config_hash"]

    manifest_path = tmp_path / "manifest.json"
    write_manifest(
        manifest_path,
        _layout(),
        records,
        source_commit="local-test",
        evaluator_version="physics-v1",
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["format"] == "chiplet-layout-mvp1-v1"
    assert manifest["input_checksum"] == _layout().checksum()
    assert manifest["record_count"] == 6
    assert manifest["source_commit"] == "local-test"


def test_sample_arms_records_paired_input_and_delta_metrics():
    records = sample_arms(
        _layout(),
        default_arm_catalog(),
        workers={
            "sa": SAWorker(),
            "analytical": AnalyticalWorker(),
            "rl-compatible": RLCompatibleWorker(),
        },
        evaluator=PhysicsEvaluator(),
        seeds=(3,),
    )

    record = records[0]
    assert record.input_metrics["hpwl"] == 5.0
    assert record.metrics["hpwl"] >= 0.0
    assert record.delta_metrics["hpwl"] == record.metrics["hpwl"] - 5.0
    assert record.metadata["total_evaluator_calls"] == 6
