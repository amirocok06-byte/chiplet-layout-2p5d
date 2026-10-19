import json
import os
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from chiplet_layout_workflow.artifacts.checksums import checksum_tree, sha256_file
from chiplet_layout_workflow.artifacts.manifest import (
    ManifestConflictError,
    InvalidStatusTransition,
    ManifestWriteError,
    transition_status,
    read_manifest,
    write_manifest,
)
from chiplet_layout_workflow.artifacts.run_store import (
    DataRootMutationLock,
    PathBoundaryError,
    RunStore,
    _is_link_or_reparse,
)
from chiplet_layout_workflow.contracts import RunManifest
from chiplet_layout_workflow.research_target import ResearchTarget, TargetDecision, TargetProposal


def _manifest(run_id: str, *, status: str = "pending", marker: str = "one") -> dict:
    return {
        "run_id": run_id,
        "checksums": {"input": marker},
        "source_commit": "commit-1",
        "config_checksum": "config-1",
        "metric_schema_checksum": "metric-1",
        "executor": "local",
        "model_provider": "mock",
        "status": status,
        "research_target": ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局"),
    }


def _approved_target() -> ResearchTarget:
    proposal = TargetProposal.from_paper_evidence(
        target_id="ai42.5Dic",
        display_name="ai42.5Dic",
        source="paper_evidence",
        evidence_checksum="approved-evidence",
    )
    return ResearchTarget.from_approval(proposal, TargetDecision.approve(proposal), revision=1)


def _create_run(store: RunStore, topic: str = "layout"):
    return store.create(topic, research_target=_approved_target())


@pytest.mark.parametrize("research_target", [None, ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局")])
def test_create_requires_approved_target_before_allocating_run(tmp_path, research_target):
    store = RunStore(tmp_path)

    with pytest.raises(ValueError, match="approved research target"):
        store.create("layout", research_target=research_target)

    assert not (tmp_path / "runs").exists()


def test_create_rejects_approved_target_without_approval_provenance(tmp_path):
    forged_target = ResearchTarget.model_construct(
        target_id="ai42.5Dic",
        display_name="ai42.5Dic",
        revision=1,
        source="user_seed",
        evidence_checksum="seed",
        approved=True,
        approval_receipt=None,
    )

    with pytest.raises(ValueError, match="approval"):
        RunStore(tmp_path).create("layout", research_target=forged_target)

    assert not (tmp_path / "runs").exists()


def test_create_superseding_rejects_approved_target_without_approval_provenance(tmp_path):
    store = RunStore(tmp_path)
    parent = _create_run(store)
    forged_target = ResearchTarget.model_construct(
        target_id="ai42.5Dic",
        display_name="ai42.5Dic",
        revision=2,
        source="user_seed",
        evidence_checksum="seed",
        approved=True,
        approval_receipt=None,
    )

    with pytest.raises(ValueError, match="approval"):
        store.create_superseding(parent, "layout", research_target=forged_target)

    assert len(list((tmp_path / "runs").iterdir())) == 1


def test_create_run_is_unique_and_creates_required_directories(tmp_path):
    store = RunStore(tmp_path)

    first = _create_run(store, "A topic / with unsafe punctuation")
    second = _create_run(store, "A topic / with unsafe punctuation")

    assert first.run_id != second.run_id
    assert first.path.parent == tmp_path / "runs"
    assert first.run_id.startswith("a-topic-with-unsafe-punctuation-")
    for name in (
        "inputs",
        "paper_understanding",
        "feasibility",
        "source",
        "verification",
        "versioning",
        "logs",
        "checkpoints",
    ):
        assert getattr(first, name).is_dir()


def test_path_guard_rejects_outside_run(tmp_path):
    run = _create_run(RunStore(tmp_path))

    assert run.assert_inside(run.inputs / "paper.md") == run.inputs / "paper.md"
    with pytest.raises(PathBoundaryError):
        run.assert_inside(tmp_path / "outside.txt")


def test_checksums_are_deterministic_and_skip_external_symlinks(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "b.txt").write_bytes(b"b")
    (root / "a.txt").write_bytes(b"a")
    outside = tmp_path / "outside.txt"
    outside.write_bytes(b"secret")
    try:
        (root / "external.txt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")

    checksums = checksum_tree(root)

    assert list(checksums) == ["a.txt", "b.txt"]
    assert checksums["a.txt"] == sha256_file(root / "a.txt")
    assert checksum_tree(root) == checksums


def test_write_manifest_is_stable_and_protects_existing_contents(tmp_path):
    run = _create_run(RunStore(tmp_path))
    manifest_path = write_manifest(run, _manifest(run.run_id))

    first_bytes = manifest_path.read_bytes()
    serialized = json.loads(first_bytes)
    expected = RunManifest.model_validate(_manifest(run.run_id)).model_dump(mode="json", by_alias=True)
    assert serialized == expected
    assert first_bytes == manifest_path.read_bytes()
    write_manifest(run, _manifest(run.run_id))
    with pytest.raises(ManifestConflictError):
        write_manifest(run, _manifest(run.run_id, marker="two"))


def test_duplicate_manifest_creation_is_race_safe(tmp_path):
    run = _create_run(RunStore(tmp_path))
    manifest = _manifest(run.run_id)

    def create_manifest():
        return write_manifest(run, manifest)

    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(executor.map(lambda _: create_manifest(), range(32)))

    assert all(path == run.path / "manifest.json" for path in paths)
    assert json.loads((run.path / "manifest.json").read_bytes())["run_id"] == run.run_id
    assert (tmp_path / ".chiplet-layout-workflow.lock").is_file()


def test_concurrent_run_creation_is_serialized_under_data_root_lock(tmp_path):
    store = RunStore(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        runs = list(executor.map(lambda _: _create_run(store), range(16)))

    assert len({run.run_id for run in runs}) == 16
    assert all(run.path.parent == tmp_path / "runs" for run in runs)


def test_create_superseding_run_uses_fresh_paths_and_records_parent(tmp_path):
    store = RunStore(tmp_path)
    parent = _create_run(store)

    child = store.create_superseding(parent, "layout", research_target=_approved_target())

    assert child.parent_run_id == parent.run_id
    assert child.path != parent.path
    assert child.source != parent.source
    assert child.verification != parent.verification
    assert child.versioning != parent.versioning


def test_create_superseding_run_rejects_reused_parent_artifacts(tmp_path):
    store = RunStore(tmp_path)
    parent = _create_run(store)

    with pytest.raises(PathBoundaryError, match="reuse"):
        store.create_superseding(
            parent,
            "layout",
            research_target=_approved_target(),
            reused_paths=[parent.source, parent.path / "metrics", parent.path / "results"],
        )


def test_link_inspection_failure_is_not_treated_as_safe(tmp_path, monkeypatch):
    target = tmp_path / "target"
    target.mkdir()

    def fail_inspection(*args, **kwargs):
        raise PermissionError("inspection denied")

    monkeypatch.setattr(type(target), "stat", fail_inspection)

    with pytest.raises(PermissionError, match="inspection denied"):
        _is_link_or_reparse(target)


def test_lock_release_survives_close_failure_and_preserves_operation_error(tmp_path, monkeypatch):
    lock = DataRootMutationLock(tmp_path)

    def fail_close():
        raise OSError("close failed")

    monkeypatch.setattr(lock, "close", fail_close)
    with pytest.raises(ValueError, match="operation failed"):
        with lock:
            raise ValueError("operation failed")

    acquired = []

    def acquire_from_other_thread():
        acquired.append(lock._thread_lock.acquire(timeout=0.5))
        if acquired[-1]:
            lock._thread_lock.release()

    thread = threading.Thread(target=acquire_from_other_thread)
    thread.start()
    thread.join()
    assert acquired == [True]


def test_successful_manifest_publication_ignores_temp_cleanup_failure(tmp_path, monkeypatch):
    run = _create_run(RunStore(tmp_path))
    original_unlink = Path.unlink

    def fail_temp_cleanup(path, *args, **kwargs):
        if path.name.startswith(".manifest.json."):
            raise PermissionError("cleanup denied")
        return original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_temp_cleanup)

    manifest_path = write_manifest(run, _manifest(run.run_id))

    assert manifest_path.is_file()
    assert json.loads(manifest_path.read_bytes())["run_id"] == run.run_id


def test_manifest_write_rejects_symlinked_parent(tmp_path):
    run = _create_run(RunStore(tmp_path))
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_parent = run.path / "linked"
    try:
        linked_parent.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory symlinks are unavailable")

    unsafe_run = replace(run, path=linked_parent)
    with pytest.raises(ManifestWriteError):
        write_manifest(unsafe_run, _manifest(run.run_id))
    assert not (outside / "manifest.json").exists()


def test_run_store_rejects_runs_root_link_outside_data_root(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    runs_root = data_root / "runs"
    try:
        if os.name == "nt":
            result = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(runs_root), str(outside)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                runs_root.symlink_to(outside, target_is_directory=True)
        else:
            runs_root.symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("directory links are unavailable")
    if not runs_root.exists():
        pytest.skip("directory links are unavailable")

    with pytest.raises(PathBoundaryError):
        _create_run(RunStore(data_root))
    assert not list(outside.iterdir())


def test_write_manifest_validates_manifest_and_run_id(tmp_path):
    run = _create_run(RunStore(tmp_path))

    with pytest.raises(ValueError):
        write_manifest(run, {"run_id": run.run_id})
    with pytest.raises(ValueError, match="run_id"):
        write_manifest(run, _manifest("different"))


def test_write_manifest_rejects_self_parent_at_workflow_boundary(tmp_path):
    run = _create_run(RunStore(tmp_path))
    manifest = _manifest(
        run.run_id,
        status="superseded",
    )
    manifest.update(parent_run_id=run.run_id, target_transition_action="supersede_run")

    with pytest.raises(ValueError, match="parent_run_id"):
        write_manifest(run, manifest)

    assert not (run.path / "manifest.json").exists()


def test_read_manifest_revalidates_existing_payload_at_workflow_boundary(tmp_path):
    run = _create_run(RunStore(tmp_path))
    manifest_path = run.path / "manifest.json"
    payload = _manifest(run.run_id, status="superseded")
    payload["research_target"] = payload["research_target"].model_dump(mode="json", by_alias=True)
    payload.update(parent_run_id=run.run_id, target_transition_action="supersede_run")
    manifest_path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="parent_run_id"):
        read_manifest(run)


@pytest.mark.parametrize(
    ("current", "next_status"),
    [("pending", "approved"), ("approved", "superseded"), ("rejected", "superseded")],
)
def test_status_transitions_allow_forward_or_explicit_superseded(current, next_status):
    assert transition_status(current, next_status) == next_status


@pytest.mark.parametrize(
    ("current", "next_status"),
    [("approved", "pending"), ("approved", "rejected"), ("failed", "approved"), ("superseded", "approved")],
)
def test_status_transitions_reject_invalid_moves(current, next_status):
    with pytest.raises(InvalidStatusTransition):
        transition_status(current, next_status)
