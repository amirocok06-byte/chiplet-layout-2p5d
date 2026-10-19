from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from chiplet_layout_workflow.agents.base import (
    AgentResult,
    AgentProtocolError,
    ArtifactBoundaryError,
    RunContext,
)
from chiplet_layout_workflow.agents.builder import BuilderAgent
from chiplet_layout_workflow.agents.reader import ReaderAgent
from chiplet_layout_workflow.agents.retriever import RetrieverAgent
from chiplet_layout_workflow.agents.verifier import VerifierAgent
from chiplet_layout_workflow.agents.version_controller import (
    VersionControllerAgent,
)
from chiplet_layout_workflow.artifacts import RunStore
from chiplet_layout_workflow.gates import HumanGate
from chiplet_layout_workflow.handoff_receipts import (
    BUILDER_APPROVAL_RECEIPT_FILENAME,
    issue_builder_approval_receipt,
    write_builder_approval_receipt,
)
from chiplet_layout_workflow.local_materials import scan_authorized_roots, stage_selected
from chiplet_layout_workflow.providers import MockProvider
from chiplet_layout_workflow.prompts import PromptRenderer
from chiplet_layout_workflow.research_target import (
    ResearchTarget,
    TargetDecision,
    TargetProposal,
)
from chiplet_layout_workflow.orchestrator import OrchestrationResult, WorkflowOrchestrator


def approved_target(target_id: str = "opaque-target") -> ResearchTarget:
    proposal = TargetProposal.from_paper_evidence(
        target_id=target_id,
        display_name="Injected research target",
        evidence_checksum="paper-checksum",
    )
    return ResearchTarget.from_approval(proposal, TargetDecision.approve(proposal))


def valid_reader_payload(target_id: str = "opaque-target") -> dict:
    return {
        "paper_understanding": "The source describes a constrained layout method.",
        "core_method": "The method extracts geometric modules and places them under constraints.",
        "candidate_plan": "Implement parsing and validation before integrating placement logic.",
        "transferable_ideas": "Make geometric invariants explicit and auditable.",
        "limitations": "The source does not establish performance on the complete target dataset.",
        "citations": [
            {
                "source": "paper.md",
                "source_kind": "reference_paper",
                "source_sha256": "a" * 64,
                "locator": "paragraph 1",
                "claim": "The method uses explicit geometric constraints.",
                "title": "Constrained layout method",
            }
        ],
        "target_assessment": {
            "current_target_id": target_id,
            "status": "uncertain",
            "rationale": "The evidence is relevant but requires human confirmation.",
            "evidence": [
                {
                    "source": "paper.md",
                    "claim": "The source objective overlaps the current target.",
                    "confidence": 0.8,
                }
            ],
            "human_questions": ["Should the current target remain approved?"],
            "proposal": {
                "target_id": "paper-proposed-target",
                "display_name": "Paper proposed target",
                "source": "paper_evidence",
                "evidence_checksum": "sha256:paper",
                "approved": False,
                "requires_human_approval": True,
                "approval_state": "pending",
            },
        },
    }


def make_run(tmp_path: Path, target: ResearchTarget | None = None):
    target = target or approved_target()
    run = RunStore(tmp_path / "data").create("test-run", research_target=target)
    (run.inputs / "paper.md").write_text("A paper-backed method.", encoding="utf-8")
    return run


def make_context(
    run,
    target: ResearchTarget | None = None,
    provider: MockProvider | None = None,
) -> RunContext:
    return RunContext(
        run=run,
        current_target=target or run.research_target,
        provider=provider or MockProvider({"ok": True}),
        metadata={
            "target_lock_id": (target or run.research_target).target_id,
            "require_target_evidence": False,
        },
    )


def make_complete_retriever_handoff(tmp_path: Path):
    target = approved_target("builder-handoff-target")
    run = RunStore(tmp_path / "data").create("builder-handoff-run", research_target=target)
    source_root = tmp_path / "authorized-materials"
    source_root.mkdir()
    (source_root / "paper.md").write_text("A paper-backed method.", encoding="utf-8")
    index = scan_authorized_roots({"d": source_root})
    stage_selected(index, index.records, run.inputs)
    reader_provider = MockProvider(valid_reader_payload(target.target_id))
    context = make_context(run, target, reader_provider)
    ReaderAgent(reader_provider).run(context)

    retriever_payload = {
        "feasibility_review": "The staged evidence supports a bounded feasibility prototype.",
        "revised_plan": "Implement parsing and validation before integration.",
        "evidence_table": "The staged paper supports the proposed validation boundary.",
        "metric_candidates": [
            {
                "name": "wirelength",
                "direction": "minimize",
                "approved": False,
                "source": "paper.md",
                "primary": True,
            }
        ],
        "decision": {
            "decision": "accept",
            "confidence": 0.9,
            "blocking_risks": [],
            "human_questions": [],
            "notes": "Proceed only after explicit human receipt.",
        },
        "target_change": None,
    }
    retriever_provider = MockProvider(retriever_payload)
    retriever_context = make_context(run, target, retriever_provider)
    RetrieverAgent(retriever_provider).run(retriever_context)
    return run, retriever_context


def test_run_context_rejects_artifacts_outside_the_current_stage(tmp_path):
    run = make_run(tmp_path)
    context = make_context(run)

    assert context.artifact_path("reading", "agent_status.json") == (
        run.paper_understanding / "agent_status.json"
    )
    with pytest.raises(ArtifactBoundaryError):
        context.artifact_path("reading", "../outside.txt")
    with pytest.raises(ArtifactBoundaryError):
        context.artifact_path("reading", tmp_path / "outside.txt")


def test_stage_bound_context_rejects_cross_stage_artifacts_and_results(tmp_path):
    run = make_run(tmp_path)
    context = make_context(run).for_stage("reading")

    with pytest.raises(ArtifactBoundaryError):
        context.artifact_root("building")
    with pytest.raises(ArtifactBoundaryError):
        context.artifact_path("building", "agent_status.json")
    with pytest.raises(ArtifactBoundaryError):
        AgentResult.from_artifact_tree(
            context,
            stage="building",
            status="scaffolded",
            created_files=(),
        )
    with pytest.raises(ArtifactBoundaryError):
        AgentResult(
            stage="building",
            status="scaffolded",
            created_files=(),
            checksums={},
        ).validate(context)


def test_agent_result_validates_created_files_and_records_protocol_fields(tmp_path):
    run = make_run(tmp_path)
    context = make_context(run)
    path = context.artifact_path("reading", "agent_status.json")
    path.write_text("{}", encoding="utf-8")

    result = AgentResult.from_artifact_tree(
        context,
        stage="reading",
        status="scaffolded",
        created_files=[path],
        warnings=["full reading execution is not implemented"],
        human_review_questions=["Approve the injected target context?"],
    )

    assert result.stage == "reading"
    assert result.status == "scaffolded"
    assert result.created_files == ("paper_understanding/agent_status.json",)
    assert result.checksums["agent_status.json"]
    assert result.warnings
    assert result.human_review_questions

    with pytest.raises(ArtifactBoundaryError):
        AgentResult(
            stage="reading",
            status="scaffolded",
            created_files=("../outside.txt",),
            checksums={},
            warnings=(),
            human_review_questions=(),
        ).validate(context)


def test_reader_requires_a_paper_or_note_and_does_not_use_a_topic_default(tmp_path):
    target = approved_target("ai42.5Dic")
    run = RunStore(tmp_path / "data").create("unrelated-run-name", research_target=target)
    provider = MockProvider(valid_reader_payload("ai42.5Dic"))
    context = make_context(run, target, provider)
    agent = ReaderAgent(provider)

    with pytest.raises(ValueError, match="paper or note"):
        agent.run(context)

    (run.inputs / "note.md").write_text("Evidence note.", encoding="utf-8")
    result = agent.run(context)
    payload = json.loads((run.paper_understanding / "agent_status.json").read_text())
    assert result.stage == "reading"
    assert payload["target"]["id"] == "ai42.5Dic"
    assert "直角多边形软模块" not in json.dumps(payload, ensure_ascii=False)


def test_explicit_input_paths_fail_closed_to_regular_files_in_run_inputs(tmp_path):
    run = make_run(tmp_path)
    outside = tmp_path / "outside.md"
    outside.write_text("outside evidence", encoding="utf-8")
    directory = run.inputs / "directory.md"
    directory.mkdir()
    missing = run.inputs / "missing.md"

    for path in (outside, directory, missing):
        with pytest.raises(ArtifactBoundaryError):
            replace(make_context(run), input_paths=(path,))

    valid_context = replace(
        make_context(run), input_paths=(run.inputs / "paper.md",)
    )
    assert valid_context.evidence_paths() == (run.inputs / "paper.md",)


def test_explicit_symlink_input_is_rejected(tmp_path):
    run = make_run(tmp_path)
    link = run.inputs / "linked.md"
    try:
        link.symlink_to(run.inputs / "paper.md")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ArtifactBoundaryError):
        replace(make_context(run), input_paths=(link,))


def test_source_worktree_is_restricted_to_source(tmp_path):
    run = make_run(tmp_path)

    assert make_context(run).source_worktree == run.source
    for path in (run.inputs, run.verification, tmp_path / "outside"):
        with pytest.raises(ArtifactBoundaryError):
            replace(make_context(run), source_worktree=path)


def test_source_worktree_rejects_symlink_components(tmp_path):
    run = make_run(tmp_path)
    target = run.source / "nested"
    target.mkdir()
    link = run.source / "linked"
    try:
        link.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    with pytest.raises(ArtifactBoundaryError):
        replace(make_context(run), source_worktree=link)


def test_reader_rejects_empty_supported_input(tmp_path):
    target = approved_target("empty-input")
    run = RunStore(tmp_path / "data").create("empty-input", research_target=target)
    (run.inputs / "empty.md").write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="paper or note"):
        ReaderAgent(MockProvider({"ok": True})).run(make_context(run, target))


def test_retriever_accepts_complete_reader_output_set_without_generating(tmp_path):
    target = approved_target("handoff-target")
    run = RunStore(tmp_path / "data").create("handoff-run", research_target=target)
    source_root = tmp_path / "authorized-materials"
    source_root.mkdir()
    (source_root / "paper.md").write_text("A paper-backed method.", encoding="utf-8")
    index = scan_authorized_roots({"d": source_root})
    stage_selected(index, index.records, run.inputs)
    provider = MockProvider(valid_reader_payload("handoff-target"))
    context = make_context(run, target, provider)

    ReaderAgent(provider).run(context)

    RetrieverAgent(MockProvider({"ok": True})).validate_inputs(context)


@pytest.mark.parametrize(
    "removed_name",
    ["transferable_ideas.md", "target_assessment.json", "reader_output_manifest.json"],
)
def test_retriever_rejects_incomplete_reader_handoff(tmp_path, removed_name):
    target = approved_target("incomplete-handoff-target")
    run = RunStore(tmp_path / "data").create(
        "incomplete-handoff-run", research_target=target
    )
    (run.inputs / "paper.md").write_text("A paper-backed method.", encoding="utf-8")
    provider = MockProvider(valid_reader_payload("incomplete-handoff-target"))
    context = make_context(run, target, provider)

    ReaderAgent(provider).run(context)
    (run.paper_understanding / removed_name).unlink()

    with pytest.raises(ValueError, match="retriever requires approved reader artifacts"):
        RetrieverAgent(MockProvider({"ok": True})).validate_inputs(context)


def test_retriever_rejects_incomplete_reader_manifest(tmp_path):
    target = approved_target("invalid-manifest-target")
    run = RunStore(tmp_path / "data").create(
        "invalid-manifest-run", research_target=target
    )
    (run.inputs / "paper.md").write_text("A paper-backed method.", encoding="utf-8")
    provider = MockProvider(valid_reader_payload("invalid-manifest-target"))
    context = make_context(run, target, provider)

    ReaderAgent(provider).run(context)
    manifest_path = run.paper_understanding / "reader_output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["substantive_outputs"].remove("limitations.md")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        RetrieverAgent(MockProvider({"ok": True})).validate_inputs(context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("agent", "retriever"),
        ("metadata_outputs", ["agent_status.json"]),
        ("response_model", "OtherOutputEnvelope"),
        ("target_id", "other-target"),
    ],
)
def test_retriever_rejects_tampered_reader_manifest_fields(tmp_path, field, value):
    target = approved_target("tampered-manifest-target")
    run = RunStore(tmp_path / "data").create(
        "tampered-manifest-run", research_target=target
    )
    (run.inputs / "paper.md").write_text("A paper-backed method.", encoding="utf-8")
    provider = MockProvider(valid_reader_payload("tampered-manifest-target"))
    context = make_context(run, target, provider)

    ReaderAgent(provider).run(context)
    manifest_path = run.paper_understanding / "reader_output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="reader output manifest"):
        RetrieverAgent(MockProvider({"ok": True})).validate_inputs(context)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("agent", "retriever"),
        ("execution", "full stage execution is intentionally deferred"),
        ("required_outputs", []),
        ("target", {"id": "other-target"}),
    ],
)
def test_retriever_rejects_tampered_reader_agent_status_fields(tmp_path, field, value):
    target = approved_target("tampered-status-target")
    run = RunStore(tmp_path / "data").create(
        "tampered-status-run", research_target=target
    )
    (run.inputs / "paper.md").write_text("A paper-backed method.", encoding="utf-8")
    provider = MockProvider(valid_reader_payload("tampered-status-target"))
    context = make_context(run, target, provider)

    ReaderAgent(provider).run(context)
    status_path = run.paper_understanding / "agent_status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status[field] = value
    status_path.write_text(json.dumps(status), encoding="utf-8")

    with pytest.raises(ValueError, match="reader agent status"):
        RetrieverAgent(MockProvider({"ok": True})).validate_inputs(context)


def test_retriever_rejects_invalid_reader_output_envelope(tmp_path):
    target = approved_target("invalid-envelope-target")
    run = RunStore(tmp_path / "data").create(
        "invalid-envelope-run", research_target=target
    )
    (run.inputs / "paper.md").write_text("A paper-backed method.", encoding="utf-8")
    provider = MockProvider(valid_reader_payload("invalid-envelope-target"))
    context = make_context(run, target, provider)

    ReaderAgent(provider).run(context)
    (run.paper_understanding / "paper_understanding.md").write_text(
        "   ", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="reader output envelope"):
        RetrieverAgent(MockProvider({"ok": True})).validate_inputs(context)


def test_builder_requires_explicit_receipt_bound_to_accepted_handoff(tmp_path):
    run, context = make_complete_retriever_handoff(tmp_path)
    agent = BuilderAgent(MockProvider({"ok": True}))
    with pytest.raises(ValueError, match="approval receipt"):
        agent.validate_inputs(context)

    gate_result = HumanGate(
        callback=lambda stage, summary: ("approve", "Reviewed retriever handoff")
    ).request("feasibility_review", "review")
    write_builder_approval_receipt(context, gate_result)
    agent.validate_inputs(context)

    receipt_path = run.feasibility / BUILDER_APPROVAL_RECEIPT_FILENAME
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["approval_status"] == "approved"
    assert receipt["human_confirmation"] == "explicit"
    assert receipt["operator_note"] == "Reviewed retriever handoff"

    receipt["receipt_checksum"] = "tampered"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="approval receipt"):
        agent.validate_inputs(context)


def test_builder_rejects_legacy_approved_marker_without_explicit_receipt(tmp_path):
    run, context = make_complete_retriever_handoff(tmp_path)
    (run.source / "builder_result.json").write_text(
        json.dumps({"approved": True}), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="approval receipt"):
        BuilderAgent(MockProvider({"ok": True})).validate_inputs(context)


def test_builder_receipt_requires_enabled_human_gate(tmp_path):
    run, context = make_complete_retriever_handoff(tmp_path)
    gate_result = HumanGate(enabled=False).request("feasibility_review", "review")

    with pytest.raises(ValueError, match="explicit human approval"):
        issue_builder_approval_receipt(context, gate_result, operator_note="forged")


@pytest.mark.parametrize(
    ("stage", "filename", "field"),
    [
        ("reading", "reader_output_manifest.json", "target_id"),
        ("feasibility_review", "retriever_output_manifest.json", "target_id"),
    ],
)
def test_builder_rejects_tampered_upstream_manifest(tmp_path, stage, filename, field):
    run, context = make_complete_retriever_handoff(tmp_path)
    root = context.artifact_root(stage)
    manifest_path = root / filename
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = "different-target"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest"):
        BuilderAgent(MockProvider({"ok": True})).validate_inputs(context)


def test_builder_rejects_receipt_bound_to_different_current_target(tmp_path):
    run, context = make_complete_retriever_handoff(tmp_path)
    gate_result = HumanGate(
        callback=lambda stage, summary: ("approve", "Reviewed retriever handoff")
    ).request("feasibility_review", "review")
    write_builder_approval_receipt(context, gate_result)

    mismatched_context = make_context(run, approved_target("different-target"))
    with pytest.raises(ValueError, match="current target"):
        BuilderAgent(MockProvider({"ok": True})).validate_inputs(mismatched_context)


def test_verifier_rejects_malformed_decision_and_builder_result(tmp_path):
    run = make_run(tmp_path)
    context = make_context(run)
    decision = run.feasibility / "decision.json"
    builder_result = run.source / "builder_result.json"
    decision.write_text("[]", encoding="utf-8")
    builder_result.write_text(json.dumps({"approved": True}), encoding="utf-8")

    with pytest.raises(ValueError, match="valid feasibility decision"):
        VerifierAgent(MockProvider({"ok": True})).validate_inputs(context)

    decision.write_text(
        json.dumps({"decision": "accept", "confidence": 0.9}), encoding="utf-8"
    )
    builder_result.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="approved builder result"):
        VerifierAgent(MockProvider({"ok": True})).validate_inputs(context)


def test_verifier_rejects_non_mapping_and_invalid_metric_yaml(tmp_path):
    run = make_run(tmp_path)
    context = make_context(run)
    (run.feasibility / "decision.json").write_text(
        json.dumps({"decision": "accept", "confidence": 0.9}), encoding="utf-8"
    )
    (run.source / "builder_result.json").write_text(
        json.dumps({"approved": True}), encoding="utf-8"
    )
    candidates = run.feasibility / "metric_candidates.yaml"
    approved = run.verification / "approved_metrics.yaml"
    approved.write_text(
        "metrics:\n  - name: wirelength\n    direction: minimize\n    approved: true\n",
        encoding="utf-8",
    )

    candidates.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping with a metrics list"):
        VerifierAgent(MockProvider({"ok": True})).validate_inputs(context)

    candidates.write_text(
        "metrics:\n  - name: wirelength\n    direction: invalid\n    approved: false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="valid metric"):
        VerifierAgent(MockProvider({"ok": True})).validate_inputs(context)

    candidates.write_text(
        "metrics:\n  - name: wirelength\n    direction: minimize\n    approved: false\n",
        encoding="utf-8",
    )
    approved.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping with a metrics list"):
        VerifierAgent(MockProvider({"ok": True})).validate_inputs(context)


@pytest.mark.parametrize(
    ("field", "value"),
    (("direction", "maximize"), ("source", "other-paper"), ("primary", True)),
)
def test_verifier_requires_approved_metric_definition_to_match_paper_candidate(
    tmp_path, field, value
):
    run = make_run(tmp_path)
    context = make_context(run)
    (run.feasibility / "decision.json").write_text(
        json.dumps({"decision": "accept", "confidence": 0.9}), encoding="utf-8"
    )
    (run.source / "builder_result.json").write_text(
        json.dumps({"approved": True}), encoding="utf-8"
    )

    candidate = {
        "name": "wirelength",
        "direction": "minimize",
        "source": "paper",
        "primary": False,
        "approved": False,
    }
    approved = candidate | {field: value, "approved": True}
    candidates = run.feasibility / "metric_candidates.yaml"
    candidates.write_text(
        "metrics:\n"
        "  - name: wirelength\n"
        "    direction: minimize\n"
        "    source: paper\n"
        "    primary: false\n"
        "    approved: false\n",
        encoding="utf-8",
    )
    (run.verification / "approved_metrics.yaml").write_text(
        "metrics:\n"
        f"  - name: {approved['name']}\n"
        f"    direction: {approved['direction']}\n"
        f"    source: {approved['source']}\n"
        f"    primary: {'true' if approved['primary'] else 'false'}\n"
        "    approved: true\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="every approved metric"):
        VerifierAgent(MockProvider({"ok": True})).validate_inputs(context)


def test_retriever_decision_and_target_change_are_validated_without_application():
    target = approved_target("current-target")
    agent = RetrieverAgent(MockProvider({"ok": True}))

    assert agent.validate_decision({"decision": "accept", "confidence": 0.8}).decision == "accept"
    with pytest.raises(ValueError):
        agent.validate_decision({"decision": "maybe", "confidence": 0.8})
    proposal = agent.validate_target_change(
        {
            "id": "paper-target",
            "display_name": "Paper target",
            "source": "paper",
            "evidence_checksum": "evidence",
        },
        target,
    )
    assert proposal.target_id == "paper-target"


def test_version_controller_only_delegates_validated_run_paths(tmp_path):
    run = make_run(tmp_path)
    calls = []

    class FakeGitAdapter:
        def archive(self, artifact_paths, metadata):
            calls.append((artifact_paths, metadata))
            return {"status": "prepared"}

    context = RunContext(
        run=run,
        current_target=run.research_target,
        provider=MockProvider({"ok": True}),
        git_adapter=FakeGitAdapter(),
        metadata={
            "target_lock_id": run.research_target.target_id,
            "require_target_evidence": False,
        },
    )
    agent = VersionControllerAgent(MockProvider({"ok": True}))
    assert agent.archive_request(context, [run.source / "file.py"], {"run_id": run.run_id}) == {
        "status": "prepared"
    }
    assert calls[0][0] == (run.source / "file.py",)
    with pytest.raises(ValueError):
        agent.archive_request(context, [tmp_path / "outside.py"], {})


@pytest.mark.parametrize(
    "agent_cls,stage",
    [
        (ReaderAgent, "reading"),
        (RetrieverAgent, "feasibility_review"),
        (BuilderAgent, "building"),
        (VerifierAgent, "verification"),
        (VersionControllerAgent, "version_archiving"),
    ],
)
def test_each_agent_declares_a_role_and_stage_boundary(agent_cls, stage):
    agent = agent_cls(MockProvider({"ok": True}))
    assert agent.name
    assert agent.role
    assert agent.stage == stage
    assert agent.required_outputs


@pytest.mark.parametrize(
    "agent_cls,role",
    [
        (ReaderAgent, "reader"),
        (RetrieverAgent, "retriever"),
        (BuilderAgent, "builder"),
        (VerifierAgent, "verifier"),
        (VersionControllerAgent, "version_controller"),
    ],
)
def test_packaged_role_prompts_list_declared_output_names(agent_cls, role):
    prompt = PromptRenderer().render(role, {})

    for declared_output in agent_cls.required_outputs:
        output_name = declared_output.split(" (", 1)[0]
        assert output_name in prompt


def test_orchestrator_requests_a_gate_and_pauses_on_non_approval(tmp_path):
    run = make_run(tmp_path)
    reader_provider = MockProvider(valid_reader_payload())
    gate_calls: list[tuple[str, str]] = []
    gate = HumanGate(
        callback=lambda stage, summary: gate_calls.append((stage, summary)) or "revise"
    )
    orchestrator = WorkflowOrchestrator(
        run,
        current_target=run.research_target,
        reader=ReaderAgent(reader_provider),
        gate=gate,
        target_lock_id=run.research_target.target_id,
        require_target_evidence=False,
    )

    result = orchestrator.run()

    assert isinstance(result, OrchestrationResult)
    assert result.status == "paused"
    assert result.stage == "reading"
    assert result.gate_result.decision == "revise"
    assert gate_calls and gate_calls[0][0] == "reading"
    assert not (run.checkpoints / "reading.json").exists()


def test_orchestrator_rejects_injected_agent_for_wrong_stage(tmp_path):
    run = make_run(tmp_path)

    class WrongStageAgent:
        stage = "building"

        def run(self, context):
            raise AssertionError("wrong-stage agent must not run")

    orchestrator = WorkflowOrchestrator(
        run,
        current_target=run.research_target,
        reader=WrongStageAgent(),
        gate=HumanGate(callback=lambda stage, summary: "approve"),
        target_lock_id=run.research_target.target_id,
        require_target_evidence=False,
    )

    with pytest.raises(AgentProtocolError, match="does not match current stage"):
        orchestrator.run()


def test_orchestrator_validates_agent_result_before_gate_and_checkpoint(tmp_path):
    run = make_run(tmp_path)
    gate_calls: list[tuple[str, str]] = []

    class MalformedResultAgent:
        stage = "reading"

        def run(self, context):
            return AgentResult(
                stage="reading",
                status="scaffolded",
                created_files=(),
                checksums={"unexpected.txt": "not-real"},
            )

    orchestrator = WorkflowOrchestrator(
        run,
        current_target=run.research_target,
        reader=MalformedResultAgent(),
        gate=HumanGate(
            callback=lambda stage, summary: gate_calls.append((stage, summary)) or "approve"
        ),
        target_lock_id=run.research_target.target_id,
        require_target_evidence=False,
    )

    with pytest.raises(AgentProtocolError, match="checksums do not match"):
        orchestrator.run()

    assert not gate_calls
    assert not (run.checkpoints / "reading.json").exists()


def test_orchestrator_is_serial_and_uses_injected_target_and_provider(tmp_path):
    target = approved_target("paper-selected-target")
    run = make_run(tmp_path, target)
    seen: list[str] = []

    class RecordingProvider(MockProvider):
        def generate(self, system_prompt, user_prompt, artifacts, response_schema=None):
            seen.append(user_prompt)
            return super().generate(system_prompt, user_prompt, artifacts, response_schema)

    reader_provider = RecordingProvider(valid_reader_payload("paper-selected-target"))

    gate = HumanGate(callback=lambda stage, summary: "approve")
    orchestrator = WorkflowOrchestrator(
        run,
        current_target=target,
        reader=ReaderAgent(reader_provider),
        gate=gate,
        target_lock_id=target.target_id,
        require_target_evidence=False,
    )

    result = orchestrator.run()

    assert result.status == "approved"
    assert result.stage == "reading"
    assert gate.history[0].decision == "approve"
    assert seen and "paper-selected-target" in seen[0]
    assert not list((tmp_path / "data").glob("*.json"))
