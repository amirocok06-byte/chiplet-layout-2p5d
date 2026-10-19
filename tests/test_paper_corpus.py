from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import chiplet_layout_workflow.paper_corpus as paper_corpus
from chiplet_layout_workflow.local_materials import (
    scan_authorized_roots,
    stage_selected,
)


class _Page:
    def __init__(self, text: str) -> None:
        self._text = text

    def extract_text(self) -> str:
        return self._text


class _Reader:
    pages_by_name: dict[str, list[_Page]] = {}
    fail_on_name: str | None = None

    def __init__(self, path: str | Path) -> None:
        name = Path(path).name
        if name == self.fail_on_name:
            raise RuntimeError("synthetic PDF reader failure")
        self.pages = self.pages_by_name[name]


def _staged_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    include_second_pdf: bool = False,
):
    references = tmp_path / "references"
    notes = tmp_path / "AI相关论文"
    references.mkdir()
    notes.mkdir()
    first_paper = sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]
    second_paper = sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[1]
    note_name = sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]
    (references / first_paper).write_bytes(b"not-a-real-pdf")
    if include_second_pdf:
        (references / second_paper).write_bytes(b"not-a-real-pdf-two")
    (notes / note_name).write_text("AI placement note\n", encoding="utf-8")
    monkeypatch.setattr(
        paper_corpus,
        "APPROVED_SOURCE_ROOTS",
        {"reference_paper": references, "ai_placement_note": notes},
    )
    monkeypatch.setattr(
        paper_corpus,
        "APPROVED_REFERENCE_PAPERS",
        frozenset({first_paper, second_paper} if include_second_pdf else {first_paper}),
    )
    monkeypatch.setattr(
        paper_corpus,
        "APPROVED_AI_PLACEMENT_NOTES",
        frozenset({note_name}),
    )
    monkeypatch.setattr(
        paper_corpus,
        "APPROVED_SOURCE_SHA256",
        {
            first_paper: hashlib.sha256((references / first_paper).read_bytes()).hexdigest(),
            **(
                {second_paper: hashlib.sha256((references / second_paper).read_bytes()).hexdigest()}
                if include_second_pdf
                else {}
            ),
            note_name: hashlib.sha256((notes / note_name).read_bytes()).hexdigest(),
        },
    )

    index = scan_authorized_roots({"reference": references, "ai": notes})
    selected = [record for record in index.records if record.safe_to_stage]
    run_inputs = tmp_path / "run" / "inputs"
    manifest = stage_selected(index, selected, run_inputs)
    return run_inputs, manifest


def _rewrite_corpus_manifest(path: Path, payload: dict) -> None:
    unsigned = dict(payload)
    unsigned.pop("corpus_checksum", None)
    canonical = json.dumps(
        unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    payload["corpus_checksum"] = hashlib.sha256(canonical).hexdigest()
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def test_prepare_is_deterministic_and_reopens_with_verified_hashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch)
    _Reader.pages_by_name = {
        sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]: [_Page("page one\n"), _Page("page two\n")],
    }
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)

    corpus_manifest = paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),),
        ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
    )

    reopened = paper_corpus.read_ai42_5dic_corpus_manifest(
        run_inputs / paper_corpus.CORPUS_MANIFEST_NAME, run_inputs
    )
    assert reopened == corpus_manifest
    assert corpus_manifest.target_id == "ai42.5Dic"
    assert corpus_manifest.papers[0].page_count == 2
    assert corpus_manifest.papers[0].text_parts
    assert all(part.page_start >= 1 for part in corpus_manifest.papers[0].text_parts)
    assert paper_corpus.reader_artifacts_from_corpus(
        run_inputs / paper_corpus.CORPUS_MANIFEST_NAME, run_inputs
    )


def test_corpus_checksum_is_stable_across_staging_locations_and_reasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    references = tmp_path / "references"
    notes = tmp_path / "AI相关论文"
    references.mkdir()
    notes.mkdir()
    paper_name = sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]
    note_name = sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]
    paper_path = references / paper_name
    note_path = notes / note_name
    paper_path.write_bytes(b"not-a-real-pdf")
    note_path.write_text("AI placement note\n", encoding="utf-8")
    monkeypatch.setattr(
        paper_corpus,
        "APPROVED_SOURCE_ROOTS",
        {"reference_paper": references, "ai_placement_note": notes},
    )
    monkeypatch.setattr(paper_corpus, "APPROVED_REFERENCE_PAPERS", frozenset({paper_name}))
    monkeypatch.setattr(paper_corpus, "APPROVED_AI_PLACEMENT_NOTES", frozenset({note_name}))
    monkeypatch.setattr(
        paper_corpus,
        "APPROVED_SOURCE_SHA256",
        {
            paper_name: hashlib.sha256(paper_path.read_bytes()).hexdigest(),
            note_name: hashlib.sha256(note_path.read_bytes()).hexdigest(),
        },
    )
    index = scan_authorized_roots({"reference": references, "ai": notes})
    selected = [record for record in index.records if record.safe_to_stage]
    paper_keys = (("reference", paper_name),)
    note_keys = (("ai", note_name),)
    first_inputs = tmp_path / "run-a" / "inputs"
    second_inputs = tmp_path / "run-b" / "inputs"
    first_material = stage_selected(index, selected, first_inputs, staging_reason="first")
    second_material = stage_selected(index, selected, second_inputs, staging_reason="second")
    _Reader.pages_by_name = {paper_name: [_Page("paper text")]}
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)

    first = paper_corpus.prepare_ai42_5dic_corpus(
        first_material,
        first_inputs,
        reference_paper_keys=paper_keys,
        ai_placement_note_keys=note_keys,
    )
    second = paper_corpus.prepare_ai42_5dic_corpus(
        second_material,
        second_inputs,
        reference_paper_keys=paper_keys,
        ai_placement_note_keys=note_keys,
    )

    assert first.corpus_checksum == second.corpus_checksum


def test_prepare_rejects_same_named_material_from_unapproved_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    decoy_reference = tmp_path / "decoy" / "1-参考文献"
    decoy_notes = tmp_path / "decoy" / "AI相关论文"
    approved_reference = tmp_path / "approved" / "1-参考文献"
    approved_notes = tmp_path / "approved" / "AI相关论文"
    decoy_reference.mkdir(parents=True)
    decoy_notes.mkdir(parents=True)
    approved_reference.mkdir(parents=True)
    approved_notes.mkdir(parents=True)
    paper_name = sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]
    note_name = sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]
    (decoy_reference / paper_name).write_bytes(b"decoy")
    (decoy_notes / note_name).write_text("decoy", encoding="utf-8")
    monkeypatch.setattr(
        paper_corpus,
        "APPROVED_SOURCE_ROOTS",
        {"reference_paper": approved_reference, "ai_placement_note": approved_notes},
    )
    monkeypatch.setattr(paper_corpus, "APPROVED_REFERENCE_PAPERS", frozenset({paper_name}))
    monkeypatch.setattr(paper_corpus, "APPROVED_AI_PLACEMENT_NOTES", frozenset({note_name}))
    index = scan_authorized_roots({"decoy-paper": decoy_reference, "decoy-note": decoy_notes})
    material_manifest = stage_selected(
        index,
        [record for record in index.records if record.safe_to_stage],
        tmp_path / "run" / "inputs",
    )

    with pytest.raises(paper_corpus.PaperCorpusError, match="outside the approved"):
        paper_corpus.prepare_ai42_5dic_corpus(
            material_manifest,
            tmp_path / "run" / "inputs",
            reference_paper_keys=(("decoy-paper", paper_name),),
            ai_placement_note_keys=(("decoy-note", note_name),),
        )


def test_prepare_rejects_incomplete_approved_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_inputs, material_manifest = _staged_run(
        tmp_path, monkeypatch, include_second_pdf=True
    )
    with pytest.raises(paper_corpus.PaperCorpusError, match="complete approved corpus"):
        paper_corpus.prepare_ai42_5dic_corpus(
            material_manifest,
            run_inputs,
            reference_paper_keys=(("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),),
            ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
        )


def test_prepare_rejects_source_hash_outside_approved_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch)
    paper_name = sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]
    note_name = sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]
    hashes = dict(paper_corpus.APPROVED_SOURCE_SHA256)
    hashes[paper_name] = "0" * 64
    monkeypatch.setattr(paper_corpus, "APPROVED_SOURCE_SHA256", hashes)

    with pytest.raises(paper_corpus.PaperCorpusError, match="outside the approved"):
        paper_corpus.prepare_ai42_5dic_corpus(
            material_manifest,
            run_inputs,
            reference_paper_keys=(("reference", paper_name),),
            ai_placement_note_keys=(("ai", note_name),),
        )


def test_surrogate_text_is_replaced_and_counted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch)
    _Reader.pages_by_name = {sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]: [_Page("before\ud800after")]}
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)

    manifest = paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),),
        ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
    )

    assert manifest.papers[0].unicode_replacement_count == 1


def test_pdf_nul_text_is_normalized_before_manifest_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch)
    paper_name = sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]
    note_name = sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]
    _Reader.pages_by_name = {paper_name: [_Page("before\x00after")]}
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)

    manifest = paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", paper_name),),
        ai_placement_note_keys=(("ai", note_name),),
    )
    reopened = paper_corpus.read_ai42_5dic_corpus_manifest(
        run_inputs / paper_corpus.CORPUS_MANIFEST_NAME, run_inputs
    )
    part = run_inputs / reopened.papers[0].text_parts[0].relative_path
    assert b"\x00" not in part.read_bytes()
    assert "�" in part.read_text(encoding="utf-8")
    assert manifest.papers[0].unicode_replacement_count == 1


def test_reader_failure_publishes_no_derived_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch, include_second_pdf=True)
    _Reader.pages_by_name = {
        sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]: [_Page("ok")],
        sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[1]: [],
    }
    _Reader.fail_on_name = sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[1]
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)

    with pytest.raises(paper_corpus.PaperCorpusError):
        paper_corpus.prepare_ai42_5dic_corpus(
            material_manifest,
            run_inputs,
            reference_paper_keys=(
                ("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),
                ("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[1]),
            ),
            ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
        )

    assert not (run_inputs / "prepared_papers").exists()
    assert not (run_inputs / paper_corpus.CORPUS_MANIFEST_NAME).exists()
    assert (run_inputs / "material_manifest.json").is_file()


def test_tampered_text_part_is_rejected_and_reader_artifacts_are_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch)
    _Reader.pages_by_name = {sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]: [_Page("paper text")]}
    _Reader.fail_on_name = None
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)
    manifest = paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),),
        ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
    )
    part_path = run_inputs / manifest.papers[0].text_parts[0].relative_path
    original_part = part_path.read_bytes()
    part_path.write_text("tampered", encoding="utf-8")

    with pytest.raises(paper_corpus.PaperCorpusError):
        paper_corpus.read_ai42_5dic_corpus_manifest(
            run_inputs / paper_corpus.CORPUS_MANIFEST_NAME, run_inputs
        )

    # The helper must return only the checksummed, text-facing corpus inputs.
    part_path.write_bytes(original_part)
    artifacts = paper_corpus.reader_artifacts_from_corpus(
        run_inputs / paper_corpus.CORPUS_MANIFEST_NAME, run_inputs
    )
    assert all(path.suffix.lower() != ".pdf" for path in artifacts)
    assert all(path.name != "material_manifest.json" for path in artifacts)
    assert any(path.name == paper_corpus.CORPUS_MANIFEST_NAME for path in artifacts)


def test_manifest_rejects_source_path_and_artifact_path_injection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch)
    _Reader.pages_by_name = {sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]: [_Page("paper text")]}
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)
    manifest_path = run_inputs / paper_corpus.CORPUS_MANIFEST_NAME
    paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),),
        ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["papers"][0]["source_relative_path"] = "F:/GAME/secret.pdf"
    _rewrite_corpus_manifest(manifest_path, payload)
    with pytest.raises(paper_corpus.PaperCorpusError):
        paper_corpus.read_ai42_5dic_corpus_manifest(manifest_path, run_inputs)

    # Rebuild a valid corpus, then try to make the reader return a non-corpus file.
    manifest_path.unlink()
    import shutil

    shutil.rmtree(run_inputs / "prepared_papers")
    paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),),
        ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["papers"][0]["text_parts"][0]["relative_path"] = "material_manifest.json"
    _rewrite_corpus_manifest(manifest_path, payload)
    with pytest.raises(paper_corpus.PaperCorpusError):
        paper_corpus.reader_artifacts_from_corpus(manifest_path, run_inputs)


def test_manifest_rechecks_text_part_size_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch)
    _Reader.pages_by_name = {sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]: [_Page("paper text")]}
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)
    manifest_path = run_inputs / paper_corpus.CORPUS_MANIFEST_NAME
    paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),),
        ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    part = payload["papers"][0]["text_parts"][0]
    oversized = b"x" * (paper_corpus.MAX_TEXT_PART_BYTES + 1)
    (run_inputs / part["relative_path"]).write_bytes(oversized)
    part["byte_size"] = len(oversized)
    part["sha256"] = hashlib.sha256(oversized).hexdigest()
    _rewrite_corpus_manifest(manifest_path, payload)

    with pytest.raises(paper_corpus.PaperCorpusError):
        paper_corpus.read_ai42_5dic_corpus_manifest(manifest_path, run_inputs)


def test_corpus_binds_material_manifest_checksum(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch)
    _Reader.pages_by_name = {sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]: [_Page("paper text")]}
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)
    manifest_path = run_inputs / paper_corpus.CORPUS_MANIFEST_NAME
    paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),),
        ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
    )

    material_path = run_inputs / "material_manifest.json"
    material_payload = json.loads(material_path.read_text(encoding="utf-8"))
    material_payload["index_checksum"] = "0" * 64
    unsigned = dict(material_payload)
    unsigned.pop("manifest_checksum", None)
    material_payload["manifest_checksum"] = hashlib.sha256(
        json.dumps(unsigned, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
    ).hexdigest()
    material_path.write_text(
        json.dumps(material_payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(paper_corpus.PaperCorpusError):
        paper_corpus.read_ai42_5dic_corpus_manifest(manifest_path, run_inputs)


def test_manifest_rejects_empty_text_part_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch)
    _Reader.pages_by_name = {sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]: [_Page("paper text")]}
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)
    manifest_path = run_inputs / paper_corpus.CORPUS_MANIFEST_NAME
    paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),),
        ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["papers"][0]["text_parts"][0]["relative_path"] = ""
    _rewrite_corpus_manifest(manifest_path, payload)

    with pytest.raises(paper_corpus.PaperCorpusError):
        paper_corpus.read_ai42_5dic_corpus_manifest(manifest_path, run_inputs)


def test_manifest_rejects_staged_path_not_bound_to_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_inputs, material_manifest = _staged_run(tmp_path, monkeypatch)
    _Reader.pages_by_name = {sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]: [_Page("paper text")]}
    monkeypatch.setattr(paper_corpus, "PdfReader", _Reader)
    manifest_path = run_inputs / paper_corpus.CORPUS_MANIFEST_NAME
    paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]),),
        ai_placement_note_keys=(("ai", sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]),),
    )
    original = run_inputs / "materials" / "reference" / sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]
    decoy = run_inputs / "materials" / "reference" / "decoy.pdf"
    decoy.write_bytes(original.read_bytes())
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["papers"][0]["staged_relative_path"] = "materials/reference/decoy.pdf"
    _rewrite_corpus_manifest(manifest_path, payload)

    with pytest.raises(paper_corpus.PaperCorpusError):
        paper_corpus.read_ai42_5dic_corpus_manifest(manifest_path, run_inputs)
