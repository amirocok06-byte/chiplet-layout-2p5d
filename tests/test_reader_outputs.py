import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator, validate
from jsonschema.exceptions import ValidationError as JSONSchemaValidationError
from pydantic import ValidationError

from chiplet_layout_workflow import reader_outputs as reader_outputs_module
from chiplet_layout_workflow import paper_corpus
from chiplet_layout_workflow.agents import reader as reader_module
from chiplet_layout_workflow.agents.base import AgentProtocolError, RunContext
from chiplet_layout_workflow.agents.reader import ReaderAgent
from chiplet_layout_workflow.artifacts import RunStore
from chiplet_layout_workflow.providers import MockProvider
from chiplet_layout_workflow.local_materials import scan_authorized_roots, stage_selected
from chiplet_layout_workflow.reader_outputs import (
    ReaderCitation,
    ReaderEvidence,
    ReaderOutputEnvelope,
    ReaderTargetAssessment,
    reader_response_schema,
)
from chiplet_layout_workflow.research_target import (
    ResearchTarget,
    TargetDecision,
    TargetProposal,
)


def _valid_payload() -> dict:
    return {
        "paper_understanding": "The paper studies rectilinear soft-module layouts.",
        "core_method": "The method extracts modules and places them under geometric constraints.",
        "candidate_plan": "Implement the parser first, then validate placements against the stated constraints.",
        "transferable_ideas": "Use explicit geometry invariants to make layout decisions auditable.",
        "limitations": "The source does not establish performance on our full dataset.",
        "citations": [
            {
                "source": "paper.pdf",
                "source_kind": "reference_paper",
                "source_sha256": "a" * 64,
                "locator": "p. 3",
                "claim": "The method represents each module with an axis-aligned boundary.",
                "title": "Rectilinear soft-module layout",
            }
        ],
        "target_assessment": {
            "current_target_id": "rectilinear-soft-module",
            "status": "uncertain",
            "rationale": "The evidence is relevant but does not fully confirm the current target.",
            "evidence": [
                {
                    "source": "paper.pdf",
                    "claim": "The paper's objective overlaps with the current target.",
                    "confidence": 0.8,
                }
            ],
            "human_questions": ["Should a human review the target before changing it?"],
            "proposal": {
                "target_id": "ai42.5Dic",
                "display_name": "ai42.5Dic",
                "source": "paper_evidence",
                "evidence_checksum": "sha256:abc123",
                "approved": False,
                "requires_human_approval": True,
                "approval_state": "pending",
            },
        },
    }


def _approved_target(target_id: str = "rectilinear-soft-module") -> ResearchTarget:
    proposal = TargetProposal.from_paper_evidence(
        target_id=target_id,
        display_name="Injected research target",
        evidence_checksum="paper-checksum",
    )
    return ResearchTarget.from_approval(proposal, TargetDecision.approve(proposal))


class RecordingProvider(MockProvider):
    def __init__(self, payload):
        super().__init__(payload)
        self.calls = []
        self.artifacts = []

    def generate(
        self, system_prompt, user_prompt, artifacts, response_schema=None
    ):
        self.artifacts = list(artifacts)
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "artifacts": list(artifacts),
                "response_schema": response_schema,
            }
        )
        return super().generate(
            system_prompt,
            user_prompt,
            artifacts,
            response_schema=response_schema,
        )


_UNSUPPORTED_PROVIDER_SCHEMA_KEYS = {
    "allOf",
    "if",
    "then",
    "else",
    "default",
    "title",
    "description",
    "minLength",
    "maxLength",
    "pattern",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "uniqueItems",
    "multipleOf",
    "format",
    "const",
}


def _reader_provider_response_schema():
    helper = getattr(reader_outputs_module, "reader_provider_response_schema", None)
    assert callable(helper), "reader_provider_response_schema helper is missing"
    return helper()


def _schema_nodes(node):
    if isinstance(node, dict):
        yield node
        for key, value in node.items():
            if key in {"properties", "$defs"} and isinstance(value, dict):
                for child in value.values():
                    yield from _schema_nodes(child)
            elif key not in {"required", "enum"}:
                yield from _schema_nodes(value)
    elif isinstance(node, list):
        for item in node:
            yield from _schema_nodes(item)


def _reader_context(tmp_path, provider, *, target_id="rectilinear-soft-module"):
    target = _approved_target(target_id)
    run = RunStore(tmp_path / "data").create("reader-run", research_target=target)
    evidence = run.inputs / "paper.md"
    evidence.write_text("A paper-backed method.", encoding="utf-8")
    return run, RunContext(
        run=run,
        current_target=target,
        provider=provider,
        metadata={"target_lock_id": target.target_id, "require_target_evidence": False},
    )


def _artifact_names(root):
    return {
        path.name
        for path in root.iterdir()
        if path.is_file()
    }


def test_reader_agent_publishes_complete_validated_output_set(tmp_path):
    payload = _valid_payload()
    provider = RecordingProvider(payload)
    run, context = _reader_context(tmp_path, provider)

    result = ReaderAgent(provider).run(context)
    root = run.paper_understanding

    expected_substantive = {
        "paper_understanding.md",
        "core_method.md",
        "candidate_plan.md",
        "transferable_ideas.md",
        "limitations.md",
        "citations.json",
        "target_assessment.json",
    }
    assert result.stage == "reading"
    assert result.status == "completed"
    assert _artifact_names(root) == expected_substantive | {
        "agent_status.json",
        "reader_output_manifest.json",
    }
    assert set(result.created_files) == {
        f"paper_understanding/{name}"
        for name in expected_substantive | {"agent_status.json", "reader_output_manifest.json"}
    }
    assert (root / "paper_understanding.md").read_text(encoding="utf-8") == payload[
        "paper_understanding"
    ]
    assert (root / "core_method.md").read_text(encoding="utf-8") == payload[
        "core_method"
    ]
    assert json.loads((root / "citations.json").read_text(encoding="utf-8")) == payload[
        "citations"
    ]
    assert json.loads(
        (root / "target_assessment.json").read_text(encoding="utf-8")
    )["current_target_id"] == "rectilinear-soft-module"
    assert (root / "reader_output_manifest.json").is_file()
    assert json.loads((root / "agent_status.json").read_text(encoding="utf-8"))["status"] == (
        "completed"
    )
    assert result.human_review_questions
    assert any("target" in question.lower() for question in result.human_review_questions)


def test_reader_agent_calls_provider_once_with_reader_provider_response_schema(tmp_path):
    payload = _valid_payload()
    provider = RecordingProvider(payload)
    run, context = _reader_context(tmp_path, provider)

    ReaderAgent(provider).run(context)

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["response_schema"] == _reader_provider_response_schema()
    assert call["artifacts"] == [run.inputs / "paper.md"]
    assert "rectilinear-soft-module" in call["user_prompt"]


def test_reader_agent_can_rerun_after_revision(tmp_path):
    first_provider = RecordingProvider(_valid_payload())
    run, context = _reader_context(tmp_path, first_provider)

    ReaderAgent(first_provider).run(context)

    revised_payload = _valid_payload()
    revised_payload["paper_understanding"] = "The revised reader understanding."
    second_provider = RecordingProvider(revised_payload)

    second_result = ReaderAgent(second_provider).run(context)
    second_result.validate(context)

    root = run.paper_understanding
    assert (root / "paper_understanding.md").read_text(encoding="utf-8") == (
        revised_payload["paper_understanding"]
    )
    assert (root / "reader_output_manifest.json").is_file()


def test_reader_agent_passes_only_validated_evidence(tmp_path):
    provider = RecordingProvider(_valid_payload())
    run, context = _reader_context(tmp_path, provider)
    (run.inputs / "ignored.py").write_text("print('ignored')", encoding="utf-8")
    (run.inputs / "empty.txt").write_text("", encoding="utf-8")

    ReaderAgent(provider).run(context)

    assert provider.artifacts == [run.inputs / "paper.md"]


def test_reader_agent_excludes_binary_pdf_and_unsupported_content(tmp_path):
    provider = RecordingProvider(_valid_payload())
    run, context = _reader_context(tmp_path, provider)
    (run.inputs / "binary.pdf").write_bytes(b"%PDF-1.7\x00\xff")
    (run.inputs / "ignored.bin").write_bytes(b"\x00\xff")

    ReaderAgent(provider).run(context)

    assert provider.artifacts == [run.inputs / "paper.md"]


def test_reader_agent_excludes_utf8_named_pdf(tmp_path):
    provider = RecordingProvider(_valid_payload())
    run, context = _reader_context(tmp_path, provider)
    (run.inputs / "utf8.pdf").write_text("PDF-looking untrusted text", encoding="utf-8")

    ReaderAgent(provider).run(context)

    assert provider.artifacts == [run.inputs / "paper.md"]


def test_reader_citation_requires_source_kind_and_sha256():
    payload = _valid_payload()
    payload["citations"][0].pop("source_kind", None)
    payload["citations"][0].pop("source_sha256", None)

    with pytest.raises(ValidationError):
        ReaderOutputEnvelope.model_validate(payload)


def test_reader_citation_rejects_invalid_source_sha256():
    payload = _valid_payload()
    payload["citations"][0]["source_sha256"] = "not-a-sha256"

    with pytest.raises(ValidationError):
        ReaderOutputEnvelope.model_validate(payload)


class _CorpusPage:
    def extract_text(self) -> str:
        return "corpus evidence"


class _CorpusReader:
    def __init__(self, path: str | Path) -> None:
        self.pages = [_CorpusPage()]


def test_reader_sends_only_verified_corpus_artifacts(tmp_path, monkeypatch):
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
    run = RunStore(tmp_path / "data").create(
        "corpus-reader",
        research_target=_approved_target("ai42.5Dic"),
    )
    material_manifest = stage_selected(
        index,
        [record for record in index.records if record.safe_to_stage],
        run.inputs,
    )
    monkeypatch.setattr(paper_corpus, "PdfReader", _CorpusReader)
    corpus_manifest = paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run.inputs,
        reference_paper_keys=(("reference", paper_name),),
        ai_placement_note_keys=(("ai", note_name),),
    )
    manifest_path = run.inputs / paper_corpus.CORPUS_MANIFEST_NAME
    payload = _valid_payload()
    payload["target_assessment"]["current_target_id"] = "ai42.5Dic"
    payload["target_assessment"]["proposal"]["target_id"] = "other-target"
    payload["citations"][0]["source"] = paper_name
    payload["citations"][0]["source_sha256"] = next(
        item.source_sha256
        for item in material_manifest.materials
        if item.source_relative_path == paper_name
    )
    provider = RecordingProvider(payload)
    target_proposal = TargetProposal.from_paper_evidence(
        target_id="ai42.5Dic",
        display_name="ai42.5Dic",
        evidence_checksum=corpus_manifest.corpus_checksum,
    )
    approved_target = ResearchTarget.from_approval(
        target_proposal, TargetDecision.approve(target_proposal)
    )
    context = RunContext(
        run=run,
        current_target=approved_target,
        provider=provider,
        metadata={"target_lock_id": "ai42.5Dic", "require_target_evidence": False},
    )

    ReaderAgent(provider).run(context)

    expected = paper_corpus.reader_artifacts_from_corpus(manifest_path, run.inputs)
    assert provider.artifacts == list(expected)
    assert all(path.suffix.lower() != ".pdf" for path in provider.artifacts)
    assert all(path.name != "material_manifest.json" for path in provider.artifacts)
    assert corpus_manifest.corpus_checksum

    bad_payload = _valid_payload()
    bad_payload["target_assessment"]["current_target_id"] = "ai42.5Dic"
    bad_payload["target_assessment"]["proposal"]["target_id"] = "other-target"
    bad_payload["citations"][0]["source"] = "F:/GAME/secret.pdf"
    bad_provider = RecordingProvider(bad_payload)
    bad_context = RunContext(
        run=run,
        current_target=approved_target,
        provider=bad_provider,
        metadata={"target_lock_id": "ai42.5Dic", "require_target_evidence": False},
    )
    with pytest.raises(AgentProtocolError, match="not bound to the verified corpus"):
        ReaderAgent(bad_provider).run(bad_context)


def test_reader_discovers_bound_corpus_when_input_paths_are_a_subset(
    tmp_path, monkeypatch
):
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
    run = RunStore(tmp_path / "data").create(
        "corpus-reader-subset",
        research_target=_approved_target("ai42.5Dic"),
    )
    material_manifest = stage_selected(
        index,
        [record for record in index.records if record.safe_to_stage],
        run.inputs,
    )
    monkeypatch.setattr(paper_corpus, "PdfReader", _CorpusReader)
    corpus_manifest = paper_corpus.prepare_ai42_5dic_corpus(
        material_manifest,
        run.inputs,
        reference_paper_keys=(("reference", paper_name),),
        ai_placement_note_keys=(("ai", note_name),),
    )
    part_path = run.inputs / corpus_manifest.papers[0].text_parts[0].relative_path
    payload = _valid_payload()
    payload["target_assessment"]["current_target_id"] = "ai42.5Dic"
    payload["target_assessment"]["proposal"]["target_id"] = "other-target"
    payload["citations"][0]["source"] = paper_name
    payload["citations"][0]["source_sha256"] = next(
        item.source_sha256
        for item in material_manifest.materials
        if item.source_relative_path == paper_name
    )
    provider = RecordingProvider(payload)
    target_proposal = TargetProposal.from_paper_evidence(
        target_id="ai42.5Dic",
        display_name="ai42.5Dic",
        evidence_checksum=corpus_manifest.corpus_checksum,
    )
    approved_target = ResearchTarget.from_approval(
        target_proposal, TargetDecision.approve(target_proposal)
    )
    context = RunContext(
        run=run,
        current_target=approved_target,
        provider=provider,
        input_paths=(part_path,),
        metadata={"target_lock_id": "ai42.5Dic", "require_target_evidence": False},
    )

    ReaderAgent(provider).run(context)

    assert provider.artifacts == list(
        paper_corpus.reader_artifacts_from_corpus(
            run.inputs / paper_corpus.CORPUS_MANIFEST_NAME, run.inputs
        )
    )


def test_ai42_reader_rejects_explicit_subset_without_verified_corpus(tmp_path):
    provider = RecordingProvider(_valid_payload())
    target = _approved_target("ai42.5Dic")
    run = RunStore(tmp_path / "data").create("ai42-legacy-subset", research_target=target)
    selected = run.inputs / "selected.md"
    selected.write_text("selected evidence", encoding="utf-8")
    (run.inputs / "hidden.md").write_text("hidden evidence", encoding="utf-8")
    context = RunContext(
        run=run,
        current_target=target,
        provider=provider,
        input_paths=(selected,),
        metadata={"target_lock_id": "ai42.5Dic", "require_target_evidence": False},
    )

    with pytest.raises(AgentProtocolError, match="requires a verified corpus manifest"):
        ReaderAgent(provider).run(context)


def test_reader_agent_rejects_invalid_json_without_publishing_outputs(tmp_path):
    provider = RecordingProvider("not-json")
    run, context = _reader_context(tmp_path, provider)

    with pytest.raises(AgentProtocolError):
        ReaderAgent(provider).run(context)

    assert _artifact_names(run.paper_understanding) == set()
    assert not list(run.paper_understanding.glob(".reader-*"))


def test_reader_agent_reports_incomplete_rollback_when_restore_fails(tmp_path, monkeypatch):
    first_provider = RecordingProvider(_valid_payload())
    run, context = _reader_context(tmp_path, first_provider)
    ReaderAgent(first_provider).run(context)

    second_provider = RecordingProvider(_valid_payload())
    original_replace = reader_module.os.replace
    publish_attempts = 0

    def failing_replace(source, destination):
        nonlocal publish_attempts
        source = Path(source)
        destination = Path(destination)
        if source.parent.name.startswith(".reader-"):
            publish_attempts += 1
            if publish_attempts == 2:
                raise OSError("publish failure")
        if source.parent.name == "backup" and destination.parent == run.paper_understanding:
            raise OSError("restore failure")
        return original_replace(source, destination)

    monkeypatch.setattr(reader_module.os, "replace", failing_replace)

    with pytest.raises(AgentProtocolError, match="rollback incomplete"):
        ReaderAgent(second_provider).run(context)

    staging_paths = list(run.paper_understanding.glob(".reader-*"))
    assert staging_paths
    assert (staging_paths[0] / "backup").is_dir()


def test_reader_agent_rejects_assessment_for_different_current_target(tmp_path):
    payload = _valid_payload()
    payload["target_assessment"]["current_target_id"] = "other-target"
    provider = RecordingProvider(payload)
    run, context = _reader_context(tmp_path, provider)

    with pytest.raises(AgentProtocolError, match="current target"):
        ReaderAgent(provider).run(context)

    assert _artifact_names(run.paper_understanding) == set()


def test_reader_agent_rejects_proposal_for_current_target(tmp_path):
    payload = _valid_payload()
    payload["target_assessment"]["proposal"]["target_id"] = "rectilinear-soft-module"
    provider = RecordingProvider(payload)
    run, context = _reader_context(tmp_path, provider)

    with pytest.raises(AgentProtocolError, match="proposal"):
        ReaderAgent(provider).run(context)

    assert _artifact_names(run.paper_understanding) == set()


def test_reader_agent_serializes_proposal_as_pending_and_unapproved(tmp_path):
    provider = RecordingProvider(_valid_payload())
    run, context = _reader_context(tmp_path, provider)

    ReaderAgent(provider).run(context)
    proposal = json.loads(
        (run.paper_understanding / "target_assessment.json").read_text(encoding="utf-8")
    )["proposal"]

    assert proposal["approved"] is False
    assert proposal["approval_state"] == "pending"
    assert proposal["requires_human_approval"] is True


def test_complete_reader_output_payload_is_valid():
    result = ReaderOutputEnvelope.model_validate(_valid_payload())

    assert result.target_assessment.current_target_id == "rectilinear-soft-module"
    assert result.citations[0].locator == "p. 3"
    assert result.target_assessment.proposal is not None
    assert not result.target_assessment.proposal.approved


@pytest.mark.parametrize(
    "field",
    [
        "paper_understanding",
        "core_method",
        "candidate_plan",
        "transferable_ideas",
        "limitations",
        "citations",
        "target_assessment",
    ],
)
def test_reader_output_rejects_missing_required_fields(field):
    payload = _valid_payload()
    payload.pop(field)

    with pytest.raises(ValidationError):
        ReaderOutputEnvelope.model_validate(payload)


@pytest.mark.parametrize(
    "field",
    ["paper_understanding", "core_method", "candidate_plan", "transferable_ideas", "limitations"],
)
def test_reader_output_rejects_empty_text(field):
    payload = _valid_payload()
    payload[field] = "   "

    with pytest.raises(ValidationError):
        ReaderOutputEnvelope.model_validate(payload)


def test_reader_output_rejects_empty_citations():
    payload = _valid_payload()
    payload["citations"] = []

    with pytest.raises(ValidationError):
        ReaderOutputEnvelope.model_validate(payload)


@pytest.mark.parametrize("confidence", [-0.01, 1.01])
def test_reader_evidence_rejects_confidence_outside_zero_to_one(confidence):
    with pytest.raises(ValidationError):
        ReaderEvidence(source="paper.pdf", claim="A claim", confidence=confidence)


@pytest.mark.parametrize("proposal_change", [{"approved": True}, {"approval_state": "approved"}])
def test_reader_assessment_rejects_preapproved_proposals(proposal_change):
    payload = _valid_payload()
    payload["target_assessment"]["proposal"].update(proposal_change)

    with pytest.raises(ValidationError):
        ReaderOutputEnvelope.model_validate(payload)


def test_reader_assessment_requires_proposal_when_target_needs_change():
    payload = _valid_payload()
    payload["target_assessment"]["status"] = "needs_change"
    payload["target_assessment"].pop("proposal")

    with pytest.raises(ValidationError):
        ReaderOutputEnvelope.model_validate(payload)


def test_reader_assessment_rejects_proposal_when_target_confirmed():
    payload = _valid_payload()
    payload["target_assessment"]["status"] = "confirmed"

    with pytest.raises(ValidationError):
        ReaderOutputEnvelope.model_validate(payload)


def test_reader_models_reject_extra_fields():
    payload = _valid_payload()
    payload["extra"] = "not allowed"

    with pytest.raises(ValidationError):
        ReaderOutputEnvelope.model_validate(payload)

    with pytest.raises(ValidationError):
        ReaderCitation(source="paper.pdf", locator="p. 1", claim="Claim", extra="not allowed")


def test_reader_response_schema_is_a_dict_with_all_top_level_fields():
    schema = reader_response_schema()

    assert isinstance(schema, dict)
    assert {
        "paper_understanding",
        "core_method",
        "candidate_plan",
        "transferable_ideas",
        "limitations",
        "citations",
        "target_assessment",
    } <= set(schema["properties"])


def test_reader_response_schema_exposes_target_assessment_conditions():
    assessment_schema = reader_response_schema()["$defs"]["ReaderTargetAssessment"]
    conditions = assessment_schema["allOf"]

    needs_change = next(
        condition
        for condition in conditions
        if condition["if"]["properties"]["status"]["const"] == "needs_change"
    )
    assert "proposal" in needs_change["then"]["required"]

    confirmed = next(
        condition
        for condition in conditions
        if condition["if"]["properties"]["status"]["const"] == "confirmed"
    )
    assert confirmed["then"]["properties"]["proposal"] == {"type": "null"}

    proposal_values = next(
        condition
        for condition in conditions
        if "proposal" in condition["if"].get("required", [])
    )
    proposal_schema = proposal_values["then"]["properties"]["proposal"]
    assert proposal_schema["properties"] == {
        "approved": {"const": False},
        "approval_state": {"const": "pending"},
        "requires_human_approval": {"const": True},
    }


def test_reader_response_schema_rejects_whitespace_only_paper_understanding():
    payload = _valid_payload()
    payload["paper_understanding"] = "   "

    with pytest.raises(JSONSchemaValidationError):
        validate(instance=payload, schema=reader_response_schema())


def test_reader_response_schema_rejects_whitespace_only_proposal_display_name():
    payload = _valid_payload()
    payload["target_assessment"]["proposal"]["display_name"] = "   "

    with pytest.raises(JSONSchemaValidationError):
        validate(instance=payload, schema=reader_response_schema())


def test_reader_provider_response_schema_removes_conditional_keywords():
    schema = _reader_provider_response_schema()
    conditional_keys = {"allOf", "if", "then", "else", "const"}

    assert all(not conditional_keys.intersection(node) for node in _schema_nodes(schema))


def test_reader_provider_response_schema_removes_all_unsupported_keywords():
    schema = _reader_provider_response_schema()

    assert all(
        not _UNSUPPORTED_PROVIDER_SCHEMA_KEYS.intersection(node)
        for node in _schema_nodes(schema)
    )


def test_reader_provider_response_schema_requires_every_object_property():
    schema = _reader_provider_response_schema()

    for node in _schema_nodes(schema):
        if node.get("type") == "object":
            assert set(node["required"]) == set(node["properties"])


def test_reader_provider_response_schema_passes_draft_202012_meta_validation():
    Draft202012Validator.check_schema(_reader_provider_response_schema())
