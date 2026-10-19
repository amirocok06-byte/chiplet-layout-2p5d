from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from chiplet_layout_workflow.agents.base import AgentProtocolError, RunContext
from chiplet_layout_workflow.artifacts import RunStore
from chiplet_layout_workflow.contracts import RunConfig
from chiplet_layout_workflow.orchestrator import WorkflowOrchestrator
from chiplet_layout_workflow.local_materials import scan_authorized_roots, stage_selected
from chiplet_layout_workflow import paper_corpus
from chiplet_layout_workflow.providers import MockProvider
from chiplet_layout_workflow.research_target import ResearchTarget, TargetDecision, TargetProposal
from chiplet_layout_workflow.target_policy import (
    AI42_5DIC_TARGET_ID,
    TargetPolicyError,
    require_matching_evidence,
    require_target_id,
)


def _approved_target(target_id: str, evidence_checksum: str) -> ResearchTarget:
    proposal = TargetProposal.from_paper_evidence(
        target_id=target_id,
        display_name=target_id,
        evidence_checksum=evidence_checksum,
    )
    return ResearchTarget.from_approval(proposal, TargetDecision.approve(proposal))


class _CorpusPage:
    def __init__(self, text: str) -> None:
        self.text = text

    def extract_text(self) -> str:
        return self.text


class _CorpusReader:
    def __init__(self, path: str | Path) -> None:
        self.pages = [_CorpusPage("corpus evidence")]


def _prepared_corpus(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    references = tmp_path / "references"
    notes = tmp_path / "AI相关论文"
    references.mkdir()
    notes.mkdir()
    paper_name = sorted(paper_corpus.APPROVED_REFERENCE_PAPERS)[0]
    note_name = sorted(paper_corpus.APPROVED_AI_PLACEMENT_NOTES)[0]
    (references / paper_name).write_bytes(b"not-a-real-pdf")
    (notes / note_name).write_text("AI note", encoding="utf-8")
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
            paper_name: hashlib.sha256((references / paper_name).read_bytes()).hexdigest(),
            note_name: hashlib.sha256((notes / note_name).read_bytes()).hexdigest(),
        },
    )
    index = scan_authorized_roots({"reference": references, "ai": notes})
    material_manifest = stage_selected(
        index,
        [record for record in index.records if record.safe_to_stage],
        tmp_path / "run" / "inputs",
    )
    monkeypatch.setattr(paper_corpus, "PdfReader", _CorpusReader)
    run_inputs = tmp_path / "run" / "inputs"
    manifest = paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run_inputs,
        reference_paper_keys=(("reference", paper_name),),
        ai_placement_note_keys=(("ai", note_name),),
    )
    return run_inputs, manifest


def test_ai42_5dic_is_the_canonical_locked_target():
    assert AI42_5DIC_TARGET_ID == "ai42.5Dic"
    target = _approved_target(AI42_5DIC_TARGET_ID, "a" * 64)

    assert require_target_id(target, AI42_5DIC_TARGET_ID) is target


def test_default_run_target_is_ai42_5dic(tmp_path):
    config = RunConfig.from_input({"topic": "ai42.5Dic", "data_root": tmp_path / "data"})

    assert config.research_target.target_id == AI42_5DIC_TARGET_ID
    assert config.research_target.display_name == "ai42.5Dic"


def test_target_lock_rejects_legacy_topic():
    target = _approved_target("rectilinear-soft-module", "a" * 64)

    with pytest.raises(TargetPolicyError, match="locked to 'ai42.5Dic'"):
        require_target_id(target, AI42_5DIC_TARGET_ID)


def test_target_evidence_must_match_approved_checksum(tmp_path):
    paper = tmp_path / "ai42.5Dic-paper.md"
    paper.write_text("The correct target paper.", encoding="utf-8")
    checksum = hashlib.sha256(paper.read_bytes()).hexdigest()
    target = _approved_target(AI42_5DIC_TARGET_ID, checksum)

    assert require_matching_evidence(target, [paper], allowed_root=tmp_path) == paper


def test_target_evidence_mismatch_fails_closed(tmp_path):
    legacy = tmp_path / "legacy-2p5d.md"
    legacy.write_text("Legacy 2.5D material.", encoding="utf-8")
    target = _approved_target(AI42_5DIC_TARGET_ID, "b" * 64)

    with pytest.raises(TargetPolicyError, match="no staged evidence file matches"):
        require_matching_evidence(target, [legacy], allowed_root=tmp_path)


def test_target_evidence_accepts_validated_corpus_manifest(tmp_path, monkeypatch):
    run_inputs, manifest = _prepared_corpus(tmp_path, monkeypatch)
    target = _approved_target(AI42_5DIC_TARGET_ID, manifest.corpus_checksum)
    manifest_path = run_inputs / paper_corpus.CORPUS_MANIFEST_NAME

    assert require_matching_evidence(target, [manifest_path], allowed_root=run_inputs) == manifest_path


def test_target_evidence_rejects_corpus_text_tampering(tmp_path, monkeypatch):
    run_inputs, manifest = _prepared_corpus(tmp_path, monkeypatch)
    target = _approved_target(AI42_5DIC_TARGET_ID, manifest.corpus_checksum)
    manifest_path = run_inputs / paper_corpus.CORPUS_MANIFEST_NAME
    text_part = run_inputs / manifest.papers[0].text_parts[0].relative_path
    text_part.write_text("tampered", encoding="utf-8")

    with pytest.raises(TargetPolicyError, match="no staged evidence file matches"):
        require_matching_evidence(target, [manifest_path], allowed_root=run_inputs)


def test_malformed_corpus_manifest_cannot_fallback_to_raw_file_hash(tmp_path):
    manifest_path = tmp_path / paper_corpus.CORPUS_MANIFEST_NAME
    manifest_path.write_text("{}", encoding="utf-8")
    target = _approved_target(
        AI42_5DIC_TARGET_ID,
        hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
    )

    with pytest.raises(TargetPolicyError, match="no staged evidence file matches"):
        require_matching_evidence(target, [manifest_path], allowed_root=tmp_path)


def test_run_context_blocks_locked_reader_before_provider_execution(tmp_path):
    legacy = tmp_path / "legacy-2p5d.md"
    legacy.write_text("Legacy 2.5D material.", encoding="utf-8")
    target = _approved_target(AI42_5DIC_TARGET_ID, "c" * 64)
    run = RunStore(tmp_path / "data").create("locked", research_target=target)
    (run.inputs / legacy.name).write_bytes(legacy.read_bytes())

    with pytest.raises(AgentProtocolError, match="no staged evidence file matches"):
        RunContext(
            run=run,
            current_target=target,
            provider=MockProvider({}),
            metadata={
                "target_lock_id": AI42_5DIC_TARGET_ID,
                "require_target_evidence": True,
            },
        )


def test_run_context_defaults_to_ai42_5dic_lock(tmp_path):
    target = _approved_target("rectilinear-soft-module", "d" * 64)
    run = RunStore(tmp_path / "data").create("locked", research_target=target)

    with pytest.raises(AgentProtocolError, match="locked to 'ai42.5Dic'"):
        RunContext(run=run, current_target=target, provider=MockProvider({}))


def test_orchestrator_context_options_cannot_disable_default_evidence_gate(tmp_path):
    target = _approved_target(AI42_5DIC_TARGET_ID, "e" * 64)
    run = RunStore(tmp_path / "data").create("locked", research_target=target)

    with pytest.raises(AgentProtocolError, match="no staged evidence file matches"):
        WorkflowOrchestrator(
            run,
            provider=MockProvider({}),
            context_options={"metadata": {"require_target_evidence": False}},
        )


def test_matching_evidence_outside_allowed_root_is_rejected(tmp_path):
    allowed_root = tmp_path / "allowed"
    allowed_root.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("The correct target paper.", encoding="utf-8")
    checksum = hashlib.sha256(outside.read_bytes()).hexdigest()
    target = _approved_target(AI42_5DIC_TARGET_ID, checksum)

    with pytest.raises(TargetPolicyError, match="no staged evidence file matches"):
        require_matching_evidence(target, [outside], allowed_root=allowed_root)


def test_target_evidence_scan_uses_all_staged_inputs_not_provider_subset(tmp_path):
    paper = tmp_path / "ai42.5Dic-paper.md"
    paper.write_text("The correct target paper.", encoding="utf-8")
    checksum = hashlib.sha256(paper.read_bytes()).hexdigest()
    target = _approved_target(AI42_5DIC_TARGET_ID, checksum)
    run = RunStore(tmp_path / "data").create("locked", research_target=target)
    staged_paper = run.inputs / paper.name
    staged_paper.write_bytes(paper.read_bytes())
    decoy = run.inputs / "decoy.md"
    decoy.write_text("A non-target note.", encoding="utf-8")

    context = RunContext(
        run=run,
        current_target=target,
        provider=MockProvider({}),
        input_paths=(decoy,),
    )

    assert context.evidence_paths() == (decoy,)


def test_orchestrator_rejects_non_ai42_target_by_default(tmp_path):
    target = _approved_target("rectilinear-soft-module", "a" * 64)
    run = RunStore(tmp_path / "data").create("legacy", research_target=target)

    with pytest.raises(AgentProtocolError, match="locked to 'ai42.5Dic'"):
        WorkflowOrchestrator(run, provider=MockProvider({}))
