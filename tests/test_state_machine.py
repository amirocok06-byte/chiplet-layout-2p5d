import json
import os
import subprocess
from pathlib import Path

import pytest

import chiplet_layout_workflow.checkpoints as checkpoints_module
from chiplet_layout_workflow.artifacts import PathBoundaryError, RunStore
from chiplet_layout_workflow.checkpoints import (
    CheckpointIntegrityError,
    read_all_checkpoints,
    read_checkpoint,
    write_checkpoint,
)
from chiplet_layout_workflow.gates.human_gate import HumanGate
from chiplet_layout_workflow.state_machine import (
    InvalidTransition,
    STAGES,
    WorkflowStateMachine,
)
from chiplet_layout_workflow.research_target import (
    ResearchTarget,
    TargetDecision,
    TargetProposal,
)


def _approved_target():
    proposal = TargetProposal.from_paper_evidence(
        target_id="target",
        display_name="target",
        evidence_checksum="evidence",
    )
    return ResearchTarget.from_approval(proposal, TargetDecision.approve(proposal))


def test_stage_order_is_strict_and_only_approved_stage_advances(tmp_path):
    machine = WorkflowStateMachine(tmp_path)

    machine.start()
    assert machine.current_stage == "reading"
    with pytest.raises(InvalidTransition):
        machine.start_stage("building")

    machine.complete_stage("reading", "revise", operator_note="need more evidence")
    assert machine.current_stage == "reading"
    with pytest.raises(InvalidTransition):
        machine.advance()

    machine.complete_stage("reading", "approve")
    assert machine.current_stage == "feasibility_review"
    with pytest.raises(InvalidTransition):
        machine.complete_stage("building", "approve")

    machine.complete_stage("feasibility_review", "reject")
    assert machine.current_stage == "feasibility_review"
    with pytest.raises(InvalidTransition):
        machine.advance()


def test_state_machine_has_exactly_the_five_public_stages():
    assert STAGES == (
        "reading",
        "feasibility_review",
        "building",
        "verification",
        "version_archiving",
    )


def test_human_gate_accepts_callback_decisions_and_rejects_invalid_values():
    calls = []
    gate = HumanGate(
        callback=lambda stage, summary: calls.append((stage, summary)) or "approve"
    )

    result = gate.request("reading", "two artifacts are ready")

    assert result == "approve"
    assert result.decision == "approve"
    assert result.status == "approved"
    assert calls == [("reading", "two artifacts are ready")]

    invalid_gate = HumanGate(callback=lambda stage, summary: "maybe")
    with pytest.raises(ValueError, match="approve, revise, or reject"):
        invalid_gate.request("reading", "summary")


def test_disabled_human_gate_is_explicit_and_never_approves(tmp_path):
    callback_called = False

    def callback(stage, summary):
        nonlocal callback_called
        callback_called = True
        return "approve"

    gate = HumanGate(enabled=False, callback=callback)
    result = gate.request("reading", "summary")

    assert result == "disabled"
    assert result.decision is None
    assert result.enabled is False
    assert result.status == "disabled"
    assert gate.last_result is result
    assert callback_called is False

    machine = WorkflowStateMachine(tmp_path, gate=gate)
    machine.start()
    gate_result = machine.request_gate("summary")
    assert gate_result.status == "disabled"
    assert machine.current_stage == "reading"
    with pytest.raises(InvalidTransition, match="disabled human gate"):
        machine.complete_stage("reading", gate_result, artifact_checksums={})


@pytest.mark.parametrize("decision", ["approve", "approved"])
def test_disabled_gate_cannot_be_bypassed_by_string_approval(tmp_path, decision):
    gate = HumanGate(enabled=False)
    machine = WorkflowStateMachine(tmp_path, gate=gate)
    machine.start()

    assert machine.request_gate("summary").status == "disabled"

    with pytest.raises(InvalidTransition, match="disabled human gate"):
        machine.complete_stage("reading", decision, artifact_checksums={})

    assert machine.current_stage == "reading"
    assert machine.completed_stages == ()
    assert not (tmp_path / "checkpoints" / "reading.json").exists()


def test_gate_manifest_metadata_explicitly_records_disabled_state(tmp_path):
    gate = HumanGate(enabled=False)
    machine = WorkflowStateMachine(tmp_path, gate=gate)
    machine.start()

    assert gate.manifest_metadata() == {"human_gates_enabled": False}
    assert machine.manifest_metadata() == {"human_gates_enabled": False}


def test_checkpoint_is_atomic_and_records_required_audit_fields(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    replacements = []
    original_replace = __import__("os").replace

    def record_replace(source, target):
        replacements.append((Path(source), Path(target)))
        return original_replace(source, target)

    monkeypatch.setattr("chiplet_layout_workflow.checkpoints.os.replace", record_replace)

    path = write_checkpoint(
        checkpoint_dir,
        "reading",
        "approve",
        {"paper.txt": "abc123"},
        timestamp="2026-10-04T00:00:00+00:00",
        operator_note="approved after review",
    )

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert path.parent == checkpoint_dir
    assert payload == {
        "artifact_checksums": {"paper.txt": "abc123"},
        "decision": "approve",
        "operator_note": "approved after review",
        "stage": "reading",
        "timestamp": "2026-10-04T00:00:00+00:00",
    }
    assert replacements
    assert replacements[-1][0].parent == checkpoint_dir
    assert not list(checkpoint_dir.glob("*.tmp"))


def test_checkpoint_for_run_stays_inside_run_checkpoint_directory(tmp_path):
    run = RunStore(tmp_path / "data").create("layout", research_target=_approved_target())

    path = write_checkpoint(run, "reading", "approve", {})

    assert path.parent == run.checkpoints
    assert path.is_relative_to(run.path)
    with pytest.raises(PathBoundaryError):
        run.assert_inside(tmp_path / "outside.json")


def test_run_checkpoint_write_validates_directory_and_commit_paths(tmp_path, monkeypatch):
    run = RunStore(tmp_path / "data").create("layout", research_target=_approved_target())
    calls = []
    original_validate = checkpoints_module._validate_run_checkpoint_path

    def record_validate(run_value, path):
        calls.append(path)
        return original_validate(run_value, path)

    monkeypatch.setattr(checkpoints_module, "_validate_run_checkpoint_path", record_validate)

    write_checkpoint(run, "reading", "approve", {})

    assert calls[:2] == [run.checkpoints, run.checkpoints]
    assert calls[2].parent == run.checkpoints
    assert calls[2].suffix == ".tmp"
    assert calls[3] == run.checkpoints / "reading.json"


def test_run_checkpoint_write_holds_mutation_lock_through_publish_and_rechecks_parent(
    tmp_path, monkeypatch
):
    run = RunStore(tmp_path / "data").create("layout", research_target=_approved_target())
    events = []
    original_validate = checkpoints_module._validate_run_checkpoint_path
    original_mkstemp = checkpoints_module.tempfile.mkstemp
    original_replace = checkpoints_module.os.replace

    class RecordingLock:
        def __init__(self, data_root):
            events.append(("lock_init", Path(data_root)))

        def __enter__(self):
            events.append(("lock_enter",))
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            events.append(("lock_exit",))
            return False

    def record_validate(run_value, path):
        events.append(("validate", Path(path)))
        return original_validate(run_value, path)

    def record_mkstemp(*args, **kwargs):
        events.append(("mkstemp",))
        return original_mkstemp(*args, **kwargs)

    def record_replace(source, target):
        events.append(("replace", Path(source), Path(target)))
        return original_replace(source, target)

    monkeypatch.setattr(checkpoints_module, "DataRootMutationLock", RecordingLock)
    monkeypatch.setattr(checkpoints_module, "_validate_run_checkpoint_path", record_validate)
    monkeypatch.setattr(checkpoints_module.tempfile, "mkstemp", record_mkstemp)
    monkeypatch.setattr(checkpoints_module.os, "replace", record_replace)

    write_checkpoint(run, "reading", "approve", {})

    kinds = [event[0] for event in events]
    assert kinds[0:2] == ["lock_init", "lock_enter"]
    assert kinds.index("replace") < kinds.index("lock_exit")
    replace_index = kinds.index("replace")
    assert kinds[replace_index + 1 : -1] == ["validate", "validate"]
    assert events[replace_index + 1][1] == run.checkpoints / "reading.json"
    assert events[replace_index + 2][1] == run.checkpoints


def test_run_checkpoint_write_fails_closed_when_post_publish_validation_rejects(
    tmp_path, monkeypatch
):
    run = RunStore(tmp_path / "data").create("layout", research_target=_approved_target())
    target = run.checkpoints / "reading.json"
    published = False
    original_validate = checkpoints_module._validate_run_checkpoint_path
    original_replace = checkpoints_module.os.replace

    def reject_after_publish(run_value, path):
        if published and path == target:
            raise PathBoundaryError(f"rejected published checkpoint path: {path}")
        return original_validate(run_value, path)

    def publish(source, destination):
        nonlocal published
        result = original_replace(source, destination)
        published = True
        return result

    monkeypatch.setattr(checkpoints_module, "_validate_run_checkpoint_path", reject_after_publish)
    monkeypatch.setattr(checkpoints_module.os, "replace", publish)

    with pytest.raises(PathBoundaryError, match="published checkpoint"):
        write_checkpoint(run, "reading", "approve", {})

    assert target.exists()
    assert not list(run.checkpoints.glob("*.tmp"))


def test_run_checkpoint_write_transfers_descriptor_ownership_to_fdopen(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    opened = {}
    closed = []
    original_mkstemp = checkpoints_module.tempfile.mkstemp
    original_close = checkpoints_module.os.close

    def record_mkstemp(*args, **kwargs):
        descriptor, name = original_mkstemp(*args, **kwargs)
        opened["descriptor"] = descriptor
        return descriptor, name

    def record_close(descriptor):
        closed.append(descriptor)
        return original_close(descriptor)

    monkeypatch.setattr(checkpoints_module.tempfile, "mkstemp", record_mkstemp)
    monkeypatch.setattr(checkpoints_module.os, "close", record_close)
    monkeypatch.setattr(checkpoints_module, "_sync_directory", lambda path: None)

    write_checkpoint(checkpoint_dir, "reading", "approve", {})

    assert opened["descriptor"] not in closed


def test_run_checkpoint_write_cleans_temp_when_fsync_fails(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"

    def fail_fsync(descriptor):
        raise OSError("fsync failed")

    monkeypatch.setattr(checkpoints_module.os, "fsync", fail_fsync)

    with pytest.raises(CheckpointIntegrityError, match="unable to write checkpoint"):
        write_checkpoint(checkpoint_dir, "reading", "approve", {})

    assert not list(checkpoint_dir.glob("*.tmp"))


def test_run_checkpoint_write_cleans_temp_when_replace_fails(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr(checkpoints_module.os, "replace", fail_replace)

    with pytest.raises(CheckpointIntegrityError, match="unable to write checkpoint"):
        write_checkpoint(checkpoint_dir, "reading", "approve", {})

    assert not list(checkpoint_dir.glob("*.tmp"))


def test_run_checkpoint_write_reports_cleanup_failure_without_primary_error(tmp_path, monkeypatch):
    checkpoint_dir = tmp_path / "checkpoints"
    original_unlink = Path.unlink

    def fail_cleanup(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise PermissionError("cleanup denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(CheckpointIntegrityError, match="clean up temporary checkpoint"):
        write_checkpoint(checkpoint_dir, "reading", "approve", {})

    assert (checkpoint_dir / "reading.json").exists()


def test_run_checkpoint_write_preserves_replace_error_when_cleanup_also_fails(
    tmp_path, monkeypatch
):
    checkpoint_dir = tmp_path / "checkpoints"
    original_unlink = Path.unlink

    def fail_replace(source, target):
        raise OSError("replace failed")

    def fail_cleanup(path, *args, **kwargs):
        if path.suffix == ".tmp":
            raise PermissionError("cleanup denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(checkpoints_module.os, "replace", fail_replace)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(CheckpointIntegrityError, match="unable to write checkpoint") as exc_info:
        write_checkpoint(checkpoint_dir, "reading", "approve", {})

    assert any("cleanup" in note for note in exc_info.value.__notes__)


@pytest.mark.parametrize("reject_call", [1, 2, 3, 4])
def test_run_checkpoint_write_fails_closed_when_path_validation_rejects(
    tmp_path, monkeypatch, reject_call
):
    run = RunStore(tmp_path / "data").create("layout", research_target=_approved_target())
    calls = []
    original_validate = checkpoints_module._validate_run_checkpoint_path

    def reject_validate(run_value, path):
        calls.append(path)
        if len(calls) == reject_call:
            raise PathBoundaryError(f"rejected checkpoint path: {path}")
        return original_validate(run_value, path)

    monkeypatch.setattr(checkpoints_module, "_validate_run_checkpoint_path", reject_validate)

    with pytest.raises(PathBoundaryError):
        write_checkpoint(run, "reading", "approve", {})

    assert not (run.checkpoints / "reading.json").exists()
    assert not list(run.checkpoints.glob("*.tmp"))


@pytest.mark.skipif(os.name != "nt", reason="junctions are Windows-specific")
def test_run_checkpoint_write_rejects_real_junction_inside_run(tmp_path):
    run = RunStore(tmp_path / "data").create("layout", research_target=_approved_target())
    alternate = run.path / "alternate-checkpoints"
    alternate.mkdir()
    run.checkpoints.rmdir()
    result = subprocess.run(
        ["cmd", "/c", f'mklink /J "{run.checkpoints}" "{alternate}"'],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        run.checkpoints.mkdir()
        pytest.skip("junction creation is unavailable")

    try:
        with pytest.raises(PathBoundaryError):
            write_checkpoint(run, "reading", "approve", {})
        assert not (alternate / "reading.json").exists()
    finally:
        run.checkpoints.unlink()


def test_path_only_checkpoint_reads_keep_existing_behavior(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    write_checkpoint(checkpoint_dir, "reading", "approve", {})

    checkpoints = read_all_checkpoints(checkpoint_dir)

    assert [checkpoint.stage for checkpoint in checkpoints] == ["reading"]


def test_run_checkpoint_read_rejects_globbed_path_outside_run(tmp_path, monkeypatch):
    run = RunStore(tmp_path / "data").create("layout", research_target=_approved_target())
    outside = tmp_path / "outside" / "reading.json"
    outside.parent.mkdir()
    outside.write_text("{}", encoding="utf-8")
    original_glob = Path.glob

    def untrusted_glob(path, pattern):
        if path == run.checkpoints and pattern == "*.json":
            return iter([outside])
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", untrusted_glob)

    with pytest.raises(PathBoundaryError):
        read_all_checkpoints(run)


@pytest.mark.parametrize("unsafe_kind", ["directory", "file"])
def test_run_checkpoint_read_rejects_link_or_reparse_paths(tmp_path, monkeypatch, unsafe_kind):
    run = RunStore(tmp_path / "data").create("layout", research_target=_approved_target())
    write_checkpoint(run, "reading", "approve", {})
    unsafe_path = run.checkpoints if unsafe_kind == "directory" else run.checkpoints / "reading.json"
    original_reject_link_components = checkpoints_module._reject_link_components

    def mark_unsafe(path):
        if path == unsafe_path:
            raise PathBoundaryError(f"link or reparse path: {path}")
        return original_reject_link_components(path)

    monkeypatch.setattr(checkpoints_module, "_reject_link_components", mark_unsafe)

    with pytest.raises(PathBoundaryError):
        read_all_checkpoints(run)


def test_resume_starts_after_last_approved_checkpoint(tmp_path):
    machine = WorkflowStateMachine(tmp_path)
    machine.start()
    reading_dir = tmp_path / "reading"
    reading_dir.mkdir()
    (reading_dir / "paper.txt").write_text("paper", encoding="utf-8")
    machine.complete_stage("reading", "approved")

    resumed = WorkflowStateMachine(tmp_path)
    resumed.resume()

    assert resumed.current_stage == "feasibility_review"
    assert resumed.completed_stages == ("reading",)


def test_resume_fails_closed_when_artifact_tree_changes(tmp_path):
    machine = WorkflowStateMachine(tmp_path)
    machine.start()
    reading_dir = tmp_path / "reading"
    reading_dir.mkdir()
    artifact = reading_dir / "paper.txt"
    artifact.write_text("original", encoding="utf-8")
    machine.complete_stage("reading", "approve")
    artifact.write_text("tampered", encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError, match="reading"):
        WorkflowStateMachine(tmp_path).resume()


def test_resume_fails_closed_for_corrupt_or_invalid_checkpoint(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "reading.json").write_text("{not json", encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError):
        WorkflowStateMachine(tmp_path).resume()

    (checkpoint_dir / "reading.json").write_text(
        json.dumps(
            {
                "stage": "building",
                "decision": "approve",
                "artifact_checksums": {},
                "timestamp": "2026-10-04T00:00:00+00:00",
                "operator_note": "",
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(CheckpointIntegrityError):
        read_checkpoint(checkpoint_dir / "reading.json")


def test_checkpoint_checksum_record_tampering_is_detected(tmp_path):
    machine = WorkflowStateMachine(tmp_path)
    machine.start()
    reading_dir = tmp_path / "reading"
    reading_dir.mkdir()
    (reading_dir / "paper.txt").write_text("paper", encoding="utf-8")
    machine.complete_stage("reading", "approve")

    path = tmp_path / "checkpoints" / "reading.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["artifact_checksums"]["paper.txt"] = "wrong"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(CheckpointIntegrityError):
        WorkflowStateMachine(tmp_path).resume()
