import hashlib
import json
import stat
from pathlib import Path

import pytest

from chiplet_layout_workflow.local_materials import (
    AuthorizedMaterialRoot,
    MaterialBoundaryError,
    MaterialRecord,
    read_material_manifest,
    scan_authorized_roots,
    stage_selected,
)


def _root(key: str, path: Path) -> AuthorizedMaterialRoot:
    return AuthorizedMaterialRoot(key=key, path=path)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_scan_classifies_pdf_papers_and_ai_placement_notes_as_safe_staged_material(tmp_path):
    research = tmp_path / "research"
    srtp = tmp_path / "srtp"
    paper = research / "1-参考文献" / "paper.pdf"
    note = srtp / "AI相关论文" / "01_agent.md"
    readme_note = srtp / "AI相关论文" / "README.md"
    script = research / "1-参考文献" / "unsafe.py"
    paper.parent.mkdir(parents=True)
    note.parent.mkdir(parents=True)
    paper.write_bytes(b"%PDF-demo")
    note.write_text("# AI placement", encoding="utf-8")
    readme_note.write_text("# AI placement readme", encoding="utf-8")
    script.write_text("print('never execute')", encoding="utf-8")

    index = scan_authorized_roots([_root("research", research), _root("srtp", srtp)])
    records = {(item.root_key, item.source_relative_path): item for item in index.records}

    assert records[("research", "1-参考文献/paper.pdf")].model_dump(
        include={"file_kind", "safe_to_stage", "text_like", "category"}
    ) == {
        "file_kind": "pdf",
        "safe_to_stage": True,
        "text_like": False,
        "category": "reference_paper",
    }
    assert records[("srtp", "AI相关论文/01_agent.md")].category == "ai_placement_note"
    assert records[("srtp", "AI相关论文/README.md")].category == "ai_placement_note"
    assert records[("research", "1-参考文献/unsafe.py")].safe_to_stage is False


def test_scan_is_deterministic_and_excludes_zip_archives(tmp_path):
    srtp = tmp_path / "srtp"
    research = tmp_path / "research"
    (srtp / "AI相关论文").mkdir(parents=True)
    (research / "benchmarks").mkdir(parents=True)

    overview = "# Overview\n\nLocal evidence.\n".encode("utf-8")
    guide = "# Project guide\n".encode("utf-8")
    (srtp / "AI相关论文" / "00_总览.md").write_bytes(overview)
    (srtp / "README.md").write_text("readme", encoding="utf-8")
    (srtp / "source.zip").write_bytes(b"not an input")
    (research / "benchmarks" / "case.blocks").write_text("NumBlocks : 1\n", encoding="utf-8")
    (research / "benchmarks" / "WL-driven.json").write_text("{}", encoding="utf-8")
    (research / "项目指南.md").write_bytes(guide)

    output = tmp_path / "run.inputs"
    first = scan_authorized_roots(
        [_root("srtp", srtp), _root("research", research)],
        index_path=output / "material_index.json",
    )
    second = scan_authorized_roots(
        [_root("srtp", srtp), _root("research", research)]
    )

    assert first.model_dump(mode="json") == second.model_dump(mode="json")
    assert (output / "material_index.json").is_file()
    assert [record.source_relative_path for record in first.records] == [
        "benchmarks/WL-driven.json",
        "benchmarks/case.blocks",
        "项目指南.md",
        "AI相关论文/00_总览.md",
        "README.md",
        "source.zip",
    ]

    records = {(record.root_key, record.source_relative_path): record for record in first.records}
    assert records[("srtp", "AI相关论文/00_总览.md")].sha256 == _sha256(overview)
    assert records[("srtp", "AI相关论文/00_总览.md")].category == "ai_placement_note"
    assert records[("srtp", "README.md")].category == "readme"
    assert records[("research", "项目指南.md")].category == "project_guide"
    assert records[("research", "benchmarks/case.blocks")].category == "benchmark_input"
    assert records[("srtp", "source.zip")].safe_to_stage is False
    assert records[("srtp", "source.zip")].exclusion_reason == "zip_archive"


@pytest.mark.parametrize("missing_kind", ["missing", "file"])
def test_scan_rejects_missing_or_non_directory_authorized_root(tmp_path, missing_kind):
    root = tmp_path / "missing"
    if missing_kind == "file":
        root.write_text("not a directory", encoding="utf-8")

    with pytest.raises(MaterialBoundaryError, match="authorized root"):
        scan_authorized_roots([_root("srtp", root)])


def test_scan_excludes_link_components_and_staging_rejects_script_like_records(tmp_path):
    root = tmp_path / "srtp"
    root.mkdir()
    (root / "allowed.md").write_text("allowed", encoding="utf-8")
    (root / "unsafe.py").write_text("print('do not run')", encoding="utf-8")
    executable = root / "executable.md"
    executable.write_text("do not stage", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    outside = tmp_path / "outside.md"
    outside.write_text("outside", encoding="utf-8")
    linked = root / "linked"
    try:
        linked.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are unavailable")

    index = scan_authorized_roots([_root("srtp", root)])
    paths = {record.source_relative_path for record in index.records}
    assert "linked" not in paths
    unsafe = next(record for record in index.records if record.source_relative_path == "unsafe.py")
    assert unsafe.safe_to_stage is False
    assert unsafe.exclusion_reason == "script_like"
    executable_record = next(
        record for record in index.records if record.source_relative_path == "executable.md"
    )
    assert executable_record.safe_to_stage is False
    assert executable_record.exclusion_reason == "executable"

    with pytest.raises(MaterialBoundaryError, match="safe to stage"):
        stage_selected(index, [unsafe], tmp_path / "run.inputs")

    forged = MaterialRecord(
        root_key="srtp",
        source_relative_path="../outside.md",
        source_path=str(outside),
        file_kind="markdown",
        byte_size=outside.stat().st_size,
        sha256=_sha256(outside.read_bytes()),
        safe_to_stage=True,
        text_like=True,
        category="paper_note",
    )
    with pytest.raises(MaterialBoundaryError, match="index"):
        stage_selected(index, [forged], tmp_path / "run.inputs")


def test_scan_marks_files_with_executable_bits_unsafe_to_stage(tmp_path):
    root = tmp_path / "research"
    root.mkdir()
    executable = root / "notes.md"
    executable.write_text("do not stage", encoding="utf-8")
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    if not executable.stat().st_mode & stat.S_IXUSR:
        pytest.skip("filesystem does not expose executable mode bits")

    record = scan_authorized_roots([_root("research", root)]).records[0]

    assert record.safe_to_stage is False
    assert record.exclusion_reason == "executable"


def test_stage_selected_copies_bytes_writes_checksummed_manifest_and_does_not_mutate_source(tmp_path):
    srtp = tmp_path / "srtp"
    research = tmp_path / "research"
    srtp.mkdir()
    research.mkdir()
    markdown = srtp / "notes" / "evidence.md"
    markdown.parent.mkdir()
    markdown.write_bytes(b"UTF-8 evidence\x00\xff")
    config = research / "case.json"
    config.write_text('{"package": "demo"}\n', encoding="utf-8")
    before = {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (markdown, config)}

    index = scan_authorized_roots([_root("srtp", srtp), _root("research", research)])
    selected = [
        record
        for record in index.records
        if record.source_relative_path in {"notes/evidence.md", "case.json"}
    ]
    run_inputs = tmp_path / "D-drive" / "runs" / "run-1" / "inputs"
    manifest = stage_selected(index, selected, run_inputs)

    staged = {
        item.source_relative_path: run_inputs / item.staged_relative_path
        for item in manifest.materials
    }
    assert staged["notes/evidence.md"].read_bytes() == markdown.read_bytes()
    assert staged["case.json"].read_bytes() == config.read_bytes()
    assert all(item.source_sha256 == _sha256(staged[item.source_relative_path].read_bytes()) for item in manifest.materials)
    assert manifest.manifest_checksum
    manifest_path = run_inputs / "material_manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["manifest_checksum"] == manifest.manifest_checksum
    assert read_material_manifest(manifest_path).model_dump(mode="json") == manifest.model_dump(mode="json")
    assert before == {path: (path.stat().st_mtime_ns, path.read_bytes()) for path in (markdown, config)}


def test_manifest_can_be_reopened_from_a_different_working_directory(tmp_path, monkeypatch):
    root = tmp_path / "research"
    root.mkdir()
    (root / "README.md").write_text("contract", encoding="utf-8")
    index = scan_authorized_roots([_root("research", root)])
    run_inputs = tmp_path / "D-drive" / "inputs"
    stage_selected(index, list(index.records), run_inputs)

    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    reopened = read_material_manifest(run_inputs / "material_manifest.json")

    assert reopened.materials[0].staged_relative_path == "materials/research/README.md"
    assert (run_inputs / reopened.materials[0].staged_relative_path).read_text(encoding="utf-8") == "contract"
