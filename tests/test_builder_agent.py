from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import chiplet_layout_workflow.agents.builder as builder_module
from chiplet_layout_workflow.agents.builder import BuilderAgent
from chiplet_layout_workflow.agents.base import RunContext
from chiplet_layout_workflow.agents.reader import ReaderAgent
from chiplet_layout_workflow.agents.retriever import RetrieverAgent
from chiplet_layout_workflow.artifacts import RunStore
from chiplet_layout_workflow.gates import HumanGate
from chiplet_layout_workflow.handoff_receipts import write_builder_approval_receipt
from chiplet_layout_workflow.local_materials import scan_authorized_roots, stage_selected
from chiplet_layout_workflow.providers import MockProvider
from chiplet_layout_workflow.research_target import ResearchTarget, TargetDecision, TargetProposal


def _approved_target(target_id: str = "builder-agent-target") -> ResearchTarget:
    proposal = TargetProposal.from_paper_evidence(
        target_id=target_id,
        display_name="Builder agent target",
        source="paper_evidence",
        evidence_checksum="paper-checksum",
    )
    return ResearchTarget.from_approval(proposal, TargetDecision.approve(proposal))


def _reader_payload(target_id: str) -> dict:
    return {
        "paper_understanding": "The staged evidence describes constrained placement.",
        "core_method": "Use explicit geometry and deterministic legalization.",
        "candidate_plan": "Parse one staged Bookshelf case and validate its geometry.",
        "transferable_ideas": "Keep legality checks auditable and independent.",
        "limitations": "The evidence does not establish a complete benchmark.",
        "citations": [
            {
                "source": "paper.md",
                "source_kind": "reference_paper",
                "source_sha256": "a" * 64,
                "locator": "paragraph 1",
                "claim": "Geometry is explicit.",
            }
        ],
        "target_assessment": {
            "current_target_id": target_id,
            "status": "uncertain",
            "rationale": "The evidence is relevant to the approved target.",
            "evidence": [{"source": "paper.md", "claim": "The scope is bounded.", "confidence": 0.8}],
            "human_questions": ["Keep the approved target for the builder run?"],
            "proposal": {
                "target_id": "paper-proposed-target",
                "display_name": "Paper proposed target",
                "source": "paper_evidence",
                "evidence_checksum": "proposal-checksum",
                "approved": False,
                "requires_human_approval": True,
                "approval_state": "pending",
            },
        },
    }


def _bookshelf_files(root: Path, *, duplicate: bool = False) -> None:
    cases = [root / "cases" / "demo"]
    if duplicate:
        cases.append(root / "cases" / "other")
    for case in cases:
        case.mkdir(parents=True, exist_ok=True)
        case.with_suffix(".blocks").write_text(
            "UCLA blocks 1.0\n"
            "NumNodes : 2\n"
            "NumTerminals : 0\n"
            "u0 2 2\n"
            "u1 1 2\n",
            encoding="utf-8",
        )
        case.with_suffix(".nets").write_text(
            "UCLA nets 1.0\n"
            "NumNets : 1\n"
            "NumPins : 2\n"
            "NetDegree : 2 n0\n"
            "u0 B\n"
            "u1 B\n",
            encoding="utf-8",
        )
        case.with_suffix(".pl").write_text(
            "UCLA pl 1.0\n"
            "u0 0 0 : N\n"
            "u1 2 0 : N\n",
            encoding="utf-8",
        )
        case.with_suffix(".power").write_text(
            "UCLA power 1.0\n"
            "u0 1.5\n"
            "u1 2.0\n",
            encoding="utf-8",
        )


def _prepared_context(tmp_path: Path, *, duplicate: bool = False) -> tuple[object, RunContext]:
    target = _approved_target()
    run = RunStore(tmp_path / "data").create("builder-agent-run", research_target=target)
    material_root = tmp_path / "authorized-materials"
    material_root.mkdir()
    (material_root / "paper.md").write_text("Evidence for the bounded prototype.", encoding="utf-8")
    _bookshelf_files(material_root, duplicate=duplicate)
    index = scan_authorized_roots({"fixture": material_root})
    stage_selected(index, index.records, run.inputs)

    reader_provider = MockProvider(_reader_payload(target.target_id))
    policy = {"target_lock_id": target.target_id, "require_target_evidence": False}
    context = RunContext(
        run=run,
        current_target=target,
        provider=reader_provider,
        metadata=policy,
    )
    ReaderAgent(reader_provider).run(context)

    retriever_provider = MockProvider(
        {
            "feasibility_review": "The staged evidence supports a bounded prototype.",
            "revised_plan": "Parse, place, measure, and report one staged case.",
            "evidence_table": "The case files are the source-backed smoke evidence.",
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
                "notes": "Build only after explicit approval.",
            },
            "target_change": None,
        }
    )
    retriever_context = RunContext(
        run=run,
        current_target=target,
        provider=retriever_provider,
        metadata=policy,
    )
    RetrieverAgent(retriever_provider).run(retriever_context)
    gate_result = HumanGate(
        callback=lambda stage, summary: ("approve", "Approved bounded Builder smoke run")
    ).request("feasibility_review", "approve builder")
    write_builder_approval_receipt(retriever_context, gate_result)
    return run, retriever_context


def _tree_bytes(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_builder_publishes_independent_source_and_one_case_smoke_outputs(tmp_path):
    run, context = _prepared_context(tmp_path)

    result = BuilderAgent(MockProvider({"must_not": "be called"})).run(context)

    assert result.status == "completed"
    assert set(result.created_files) == {
        "implementation_plan.md",
        "directory_layout.md",
        "implementation_report.md",
        "builder_result.json",
        "agent_status.json",
        "builder_output_manifest.json",
        "source/chiplet_layout_builder/__init__.py",
        "source/chiplet_layout_builder/builder_prototype.py",
    }
    required = {
        "implementation_plan.md",
        "directory_layout.md",
        "implementation_report.md",
        "builder_result.json",
        "agent_status.json",
        "builder_output_manifest.json",
    }
    assert required <= {path.name for path in run.source.iterdir()}
    assert (run.source / "source" / "chiplet_layout_builder" / "builder_prototype.py").is_file()
    builder_result = json.loads((run.source / "builder_result.json").read_text(encoding="utf-8"))
    assert builder_result["approved"] is True
    assert builder_result["smoke"]["case"] == "demo"
    assert builder_result["smoke"]["metrics"]["hpwl"] == pytest.approx(2.0)
    assert builder_result["smoke"]["checks"]["package_boundary"] is True
    assert builder_result["smoke"]["checks"]["non_overlap"] is True

    report = (run.source / "implementation_report.md").read_text(encoding="utf-8")
    assert "Evidence-backed" in report
    assert "Engineering inference" in report
    assert "thermal" in report.lower()
    assert "metadata-only" in report
    assert "deferred" in report.lower()

    generated = run.source / "source"
    subprocess.run([sys.executable, "-m", "compileall", "-q", str(generated)], check=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(generated)
    probe = (
        "from chiplet_layout_builder.builder_prototype import run_baseline; "
        "print(run_baseline(r'%s', package_width=3, package_height=2)['metrics']['hpwl'])"
        % (run.inputs / "materials" / "fixture" / "cases" / "demo.blocks")
    )
    completed = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True, env=env
    )
    assert completed.stdout.strip() == "2.0"

    manifest = json.loads((run.source / "builder_output_manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["output_checksums"]["builder_result.json"] == hashlib.sha256(
        (run.source / "builder_result.json").read_bytes()
    ).hexdigest()


def test_builder_rejects_more_than_one_staged_bookshelf_case(tmp_path):
    run, context = _prepared_context(tmp_path, duplicate=True)

    with pytest.raises(ValueError, match="exactly one staged Bookshelf case"):
        BuilderAgent(MockProvider({})).run(context)

    assert list(run.source.iterdir()) == []


def test_failed_builder_rerun_preserves_previous_valid_publication(tmp_path):
    run, context = _prepared_context(tmp_path)
    BuilderAgent(MockProvider({})).run(context)
    before = _tree_bytes(run.source)

    blocks = run.inputs / "materials" / "fixture" / "cases" / "demo.blocks"
    blocks.write_text("not a Bookshelf file\n", encoding="utf-8")

    with pytest.raises(ValueError):
        BuilderAgent(MockProvider({})).run(context)

    assert _tree_bytes(run.source) == before


def test_publication_swap_failure_restores_previous_valid_publication(tmp_path, monkeypatch):
    run, context = _prepared_context(tmp_path)
    BuilderAgent(MockProvider({})).run(context)
    before = _tree_bytes(run.source)
    original_replace = builder_module.os.replace

    def fail_new_source(source, destination):
        if (
            Path(source).name.startswith(".builder-")
            and "backup-" not in Path(source).name
            and Path(destination).name == "source"
        ):
            raise OSError("simulated publication failure")
        return original_replace(source, destination)

    monkeypatch.setattr(builder_module.os, "replace", fail_new_source)
    with pytest.raises(OSError, match="simulated publication failure"):
        BuilderAgent(MockProvider({})).run(context)

    assert _tree_bytes(run.source) == before
