from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

import chiplet_layout_workflow.agents.retriever as retriever_module
from chiplet_layout_workflow.agents.base import AgentProtocolError, RunContext
from chiplet_layout_workflow.agents.reader import ReaderAgent
from chiplet_layout_workflow.agents.retriever import RetrieverAgent
from chiplet_layout_workflow.artifacts import RunStore
from chiplet_layout_workflow.providers import LLMResponse, MockProvider
from chiplet_layout_workflow.research_target import ResearchTarget, TargetDecision, TargetProposal
from chiplet_layout_workflow.retriever_outputs import (
    RetrieverOutputEnvelope,
    retriever_provider_response_schema,
)
from chiplet_layout_workflow.local_materials import scan_authorized_roots, stage_selected


def _approved_target(target_id: str = "target-a") -> ResearchTarget:
    proposal = TargetProposal.from_paper_evidence(
        target_id=target_id,
        display_name="Target A",
        evidence_checksum="paper-checksum",
    )
    return ResearchTarget.from_approval(proposal, TargetDecision.approve(proposal))


def _reader_payload(target_id: str = "target-a") -> dict:
    return {
        "paper_understanding": "The source describes a constrained layout method.",
        "core_method": "The method uses explicit geometric constraints.",
        "candidate_plan": "Parse and validate the staged inputs first.",
        "transferable_ideas": "Keep geometric invariants explicit.",
        "limitations": "The source does not establish complete target coverage.",
        "citations": [
            {
                "source": "material.md",
                "source_kind": "reference_paper",
                "source_sha256": "a" * 64,
                "locator": "paragraph 1",
                "claim": "The source uses geometric constraints.",
            }
        ],
        "target_assessment": {
            "current_target_id": target_id,
            "status": "confirmed",
            "rationale": "The staged material is relevant to the current target.",
            "evidence": [
                {
                    "source": "material.md",
                    "claim": "The source objective overlaps the current target.",
                    "confidence": 0.8,
                }
            ],
            "human_questions": [],
            "proposal": None,
        },
    }


def _retriever_payload(*, target_change: dict | None = None) -> dict:
    return {
        "feasibility_review": "The staged evidence supports a bounded feasibility prototype.",
        "revised_plan": "Implement parsing, validation, and deterministic checks before integration.",
        "evidence_table": "| source | finding |\n|---|---|\n| material.md | geometric constraints are explicit |",
        "metric_candidates": [
            {
                "name": "wirelength",
                "direction": "minimize",
                "approved": False,
                "source": "material.md",
                "primary": True,
            }
        ],
        "decision": {
            "decision": "accept",
            "confidence": 0.8,
            "blocking_risks": [],
            "human_questions": ["Approve the feasibility handoff before building."],
            "notes": "This is a feasibility recommendation only.",
        },
        "target_change": target_change,
    }


class RecordingProvider(MockProvider):
    def __init__(self, payload):
        super().__init__(payload, model="recording-model")
        self.calls: list[dict] = []

    def generate(self, system_prompt, user_prompt, artifacts, response_schema=None):
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "artifacts": tuple(artifacts),
                "response_schema": response_schema,
            }
        )
        return LLMResponse(
            text=self._payload_text(),
            model="recording-model",
            usage={"input_tokens": 11, "output_tokens": 7},
            provider_request_id="request-1",
        )

    def _payload_text(self) -> str:
        return json.dumps(self._payload, ensure_ascii=False)


def _run_context(tmp_path: Path, provider: RecordingProvider):
    target = _approved_target()
    run = RunStore(tmp_path / "data").create("retriever-test", research_target=target)
    source_root = tmp_path / "authorized-materials"
    source_root.mkdir()
    (source_root / "material.md").write_text(
        "A staged local source describing geometric constraints.", encoding="utf-8"
    )
    index = scan_authorized_roots({"d": source_root})
    stage_selected(index, index.records, run.inputs)
    policy = {"target_lock_id": target.target_id, "require_target_evidence": False}
    ReaderAgent(MockProvider(_reader_payload())).run(
        RunContext(
            run=run,
            current_target=target,
            provider=MockProvider(_reader_payload()),
            metadata=policy,
        )
    )
    return run, RunContext(
        run=run, current_target=target, provider=provider, metadata=policy
    )


def test_retriever_envelope_rejects_missing_text_bad_decision_and_approvals():
    valid = _retriever_payload()
    assert RetrieverOutputEnvelope.model_validate(valid).decision.decision == "accept"

    missing_text = dict(valid)
    missing_text.pop("feasibility_review")
    with pytest.raises(ValidationError):
        RetrieverOutputEnvelope.model_validate(missing_text)

    bad_decision = dict(valid)
    bad_decision["decision"] = {"decision": "maybe", "confidence": 0.8}
    with pytest.raises(ValidationError):
        RetrieverOutputEnvelope.model_validate(bad_decision)

    approved_metric = json.loads(json.dumps(valid))
    approved_metric["metric_candidates"][0]["approved"] = True
    with pytest.raises(ValidationError, match="unapproved"):
        RetrieverOutputEnvelope.model_validate(approved_metric)

    approved_target = json.loads(json.dumps(valid))
    approved_target["target_change"] = {
        "id": "target-b",
        "display_name": "Target B",
        "source": "material.md",
        "evidence_checksum": "evidence-b",
        "approved": True,
        "requires_human_approval": True,
        "approval_state": "approved",
    }
    with pytest.raises(ValidationError, match="pending"):
        RetrieverOutputEnvelope.model_validate(approved_target)


def test_retriever_makes_one_structured_call_and_publishes_checked_outputs(tmp_path):
    provider = RecordingProvider(_retriever_payload())
    run, context = _run_context(tmp_path, provider)

    result = RetrieverAgent(provider).run(context)

    assert len(provider.calls) == 1
    call = provider.calls[0]
    assert call["response_schema"] == retriever_provider_response_schema()
    assert call["artifacts"]
    assert all(path.parent == run.inputs / "materials" / "d" for path in call["artifacts"] if path.name != "material_manifest.json")

    expected = {
        "feasibility_review.md",
        "revised_plan.md",
        "evidence_table.md",
        "metric_candidates.yaml",
        "decision.json",
        "agent_status.json",
        "retriever_output_manifest.json",
    }
    assert set(result.created_files) == {f"feasibility/{name}" for name in expected}
    assert set(path.name for path in run.feasibility.iterdir()) == expected
    assert yaml.safe_load((run.feasibility / "metric_candidates.yaml").read_text())[
        "metrics"
    ][0]["approved"] is False

    manifest = json.loads(
        (run.feasibility / "retriever_output_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["run_id"] == run.run_id
    assert manifest["target"]["id"] == "target-a"
    assert manifest["material_manifest_checksum"]
    assert manifest["provider"] == {
        "model": "recording-model",
        "provider_request_id": "request-1",
        "usage": {"input_tokens": 11, "output_tokens": 7},
    }
    for name, checksum in manifest["output_checksums"].items():
        assert name != "retriever_output_manifest.json"
        assert result.checksums[name] == checksum


def test_retriever_requires_staged_material_manifest_and_rejects_target_mismatch(tmp_path):
    provider = RecordingProvider(_retriever_payload())
    run, context = _run_context(tmp_path, provider)
    (run.inputs / "material_manifest.json").unlink()

    with pytest.raises(ValueError, match="material manifest"):
        RetrieverAgent(provider).run(context)
    assert not list(run.feasibility.iterdir())

    provider = RecordingProvider(
        _retriever_payload(
            target_change={
                "id": "target-a",
                "display_name": "Target A",
                "source": "material.md",
                "evidence_checksum": "other-evidence",
                "approved": False,
                "requires_human_approval": True,
                "approval_state": "pending",
            }
        )
    )
    run, context = _run_context(tmp_path / "mismatch", provider)
    with pytest.raises(AgentProtocolError, match="differ from the current target"):
        RetrieverAgent(provider).run(context)
    assert not list(run.feasibility.iterdir())


def test_invalid_rerun_keeps_previous_valid_feasibility_output(tmp_path):
    first_provider = RecordingProvider(_retriever_payload())
    run, context = _run_context(tmp_path, first_provider)
    RetrieverAgent(first_provider).run(context)
    before = {
        path.name: path.read_bytes()
        for path in run.feasibility.iterdir()
        if path.is_file()
    }

    invalid = _retriever_payload()
    invalid["metric_candidates"][0]["approved"] = True
    second_provider = RecordingProvider(invalid)
    with pytest.raises(AgentProtocolError, match="schema"):
        RetrieverAgent(second_provider).run(
            RunContext(
                run=run,
                current_target=run.research_target,
                provider=second_provider,
                metadata={
                    "target_lock_id": run.research_target.target_id,
                    "require_target_evidence": False,
                },
            )
        )

    after = {
        path.name: path.read_bytes()
        for path in run.feasibility.iterdir()
        if path.is_file()
    }
    assert after == before


def test_failed_publication_rolls_back_to_previous_valid_handoff(tmp_path, monkeypatch):
    first_provider = RecordingProvider(_retriever_payload())
    run, context = _run_context(tmp_path, first_provider)
    RetrieverAgent(first_provider).run(context)
    before = {
        path.name: path.read_bytes()
        for path in run.feasibility.iterdir()
        if path.is_file()
    }

    second_provider = RecordingProvider(_retriever_payload())
    original_replace = retriever_module.os.replace
    publication_count = 0

    def fail_second_publication(source, destination):
        nonlocal publication_count
        source = Path(source)
        destination = Path(destination)
        if source.parent.name.startswith(".retriever-") and destination.parent == run.feasibility:
            publication_count += 1
            if publication_count == 2:
                raise OSError("simulated publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(retriever_module.os, "replace", fail_second_publication)
    with pytest.raises(OSError, match="simulated publication failure"):
        RetrieverAgent(second_provider).run(
            RunContext(
                run=run,
                current_target=run.research_target,
                provider=second_provider,
                metadata={
                    "target_lock_id": run.research_target.target_id,
                    "require_target_evidence": False,
                },
            )
        )

    after = {
        path.name: path.read_bytes()
        for path in run.feasibility.iterdir()
        if path.is_file()
    }
    assert after == before
    assert not list(run.feasibility.glob(".retriever-*"))


def test_retriever_provider_schema_is_strict_and_provider_safe():
    schema = retriever_provider_response_schema()

    assert set(schema["required"]) == set(schema["properties"])
    assert schema["additionalProperties"] is False
    unsupported = {"allOf", "if", "then", "else", "const", "default", "title", "minLength", "pattern"}

    def walk(node):
        if isinstance(node, dict):
            yield node
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    assert all(not unsupported.intersection(node) for node in walk(schema))
