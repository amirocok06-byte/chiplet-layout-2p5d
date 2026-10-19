import json
from pathlib import Path

import jsonschema
import pytest
from pydantic import ValidationError

from chiplet_layout_workflow.config import load_config
from chiplet_layout_workflow.contracts import RunConfig, RunManifest
from chiplet_layout_workflow.research_target import (
    ResearchTarget,
    TargetApprovalReceipt,
    TargetDecision,
    TargetProposal,
    TargetTransition,
    transition_target,
)
from chiplet_layout_workflow.validation import (
    validate_manifest_payload,
    validate_run_payload,
    validate_target_payload,
)


def _proposal(target_id="ai42.5Dic", *, evidence_checksum="abc123"):
    return TargetProposal.from_paper_evidence(
        target_id=target_id,
        display_name=target_id,
        evidence_checksum=evidence_checksum,
    )


def _approved_target(*, proposal=None, revision=1):
    proposal = proposal or _proposal()
    return ResearchTarget.from_approval(proposal, TargetDecision.approve(proposal), revision=revision)


def _manifest_payload(**overrides):
    payload = {
        "run_id": "run-1",
        "checksums": {"input": "abc"},
        "source_commit": "commit-1",
        "config_checksum": "config-1",
        "metric_schema_checksum": "metric-1",
        "executor": "local",
        "model_provider": "mock",
        "status": "pending",
        "research_target": ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局"),
    }
    payload.update(overrides)
    return payload


def test_default_seed_is_not_final_target_and_paper_id_is_opaque():
    target = ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局")
    proposal = _proposal()

    assert target.is_seed
    assert target.target_id == "rectilinear-soft-module"
    assert proposal.target_id == "ai42.5Dic"
    assert proposal.requires_human_approval
    assert not proposal.approved


@pytest.mark.parametrize("target_id", ["", "   "])
def test_target_ids_must_be_non_empty(target_id):
    with pytest.raises(ValidationError):
        ResearchTarget(target_id=target_id, display_name="name", revision=0, source="user_seed", evidence_checksum="x")


def test_target_runtime_accepts_opaque_identifier_without_domain_restrictions():
    target = ResearchTarget(
        target_id="ai42.5Dic",
        display_name="ai42.5Dic",
        revision=0,
        source="paper_evidence",
        evidence_checksum="evidence-1",
        approved=False,
    )

    assert target.target_id == "ai42.5Dic"


def test_proposals_require_evidence_checksum():
    with pytest.raises(ValidationError):
        _proposal(evidence_checksum="")
    with pytest.raises(ValidationError):
        TargetProposal(target_id="ai42.5Dic", display_name="ai42.5Dic")


def test_approved_target_requires_bound_approval_receipt():
    with pytest.raises(ValidationError, match="approval"):
        ResearchTarget(
            target_id="ai42.5Dic",
            display_name="ai42.5Dic",
            revision=1,
            source="paper_evidence",
            evidence_checksum="abc123",
            approved=True,
        )


def test_approval_receipt_is_bound_and_rehydratable():
    proposal = _proposal()
    decision = TargetDecision.approve(proposal)
    target = ResearchTarget.from_approval(proposal, decision, revision=1)

    restored = ResearchTarget.model_validate(target.model_dump(mode="json", by_alias=True))

    assert restored.require_approved() == target
    assert restored.approval_receipt is not None
    assert restored.approval_receipt.proposal.target_id == proposal.target_id


@pytest.mark.parametrize("validator", [validate_target_payload, validate_run_payload])
def test_unified_payload_boundary_rejects_receipt_target_mismatch(validator):
    target_payload = _approved_target().model_dump(mode="json", by_alias=True)
    target_payload["source"] = "different-source"
    payload = target_payload if validator is validate_target_payload else {
        "topic": "layout",
        "data_root": "D:/runs",
        "research_target": target_payload,
    }
    schema_name = "target" if validator is validate_target_payload else "run"
    schema = json.loads(
        (Path(__file__).parents[1] / "schemas" / f"{schema_name}.schema.json").read_text(encoding="utf-8")
    )
    jsonschema.validate(payload, schema)

    with pytest.raises(ValidationError, match="approval receipt"):
        validator(payload)


def test_approval_receipt_rejects_equivalent_but_unbound_decision():
    proposal = _proposal()
    decision = TargetDecision.approve(TargetProposal.model_validate(proposal.model_dump()))

    with pytest.raises(ValueError, match="bound"):
        TargetApprovalReceipt.issue(proposal, decision)


def test_research_target_requires_revision_and_approval_metadata():
    with pytest.raises(ValidationError):
        ResearchTarget.model_validate(
            {
                "id": "ai42.5Dic",
                "display_name": "ai42.5Dic",
                "source": "paper_evidence",
                "evidence_checksum": "abc123",
            }
        )


def test_research_target_aliases_are_accepted_but_ambiguous_aliases_rejected():
    common = {
        "display_name": "ai42.5Dic",
        "revision": 0,
        "source": "user_seed",
        "evidence_checksum": "seed",
        "approved": False,
    }

    assert ResearchTarget.model_validate({**common, "id": "ai42.5Dic"}).target_id == "ai42.5Dic"
    assert ResearchTarget.model_validate({**common, "target_id": "ai42.5Dic"}).target_id == "ai42.5Dic"
    with pytest.raises(ValidationError, match="id"):
        ResearchTarget.model_validate({**common, "id": "ai42.5Dic", "target_id": "other"})


def test_unapproved_target_change_is_rejected():
    target = ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局")
    proposal = _proposal()
    decision = TargetDecision.reject(proposal)

    with pytest.raises(ValueError, match="approved"):
        transition_target(target, proposal, decision=decision, stage="feasibility_review")
    with pytest.raises(ValueError, match="approved"):
        transition_target(target, proposal, stage="feasibility_review", approved=False)


@pytest.mark.parametrize(
    "overrides",
    [
        {"target_id": "other-target"},
        {"evidence_checksum": "different-checksum"},
        {"source": "other-source"},
    ],
)
def test_target_decision_rejects_mismatched_proposal_metadata(overrides):
    proposal = _proposal()

    with pytest.raises(ValidationError, match="proposal"):
        TargetDecision(proposal=proposal, approved=True, decision="approve", **overrides)


def test_transition_rejects_decision_without_nested_proposal():
    target = ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局")
    proposal = _proposal()
    decision = TargetDecision(
        target_id=proposal.target_id,
        source=proposal.source,
        evidence_checksum=proposal.evidence_checksum,
        approved=True,
    )

    with pytest.raises(ValueError, match="proposal"):
        transition_target(target, proposal, decision=decision, stage="feasibility_review")


def test_transition_rejects_decision_bound_to_equivalent_but_different_proposal():
    target = ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局")
    proposal = _proposal()
    equivalent_proposal = TargetProposal.model_validate(proposal.model_dump())
    decision = TargetDecision.approve(equivalent_proposal)

    with pytest.raises(ValueError, match="proposal"):
        transition_target(target, proposal, decision=decision, stage="feasibility_review")


@pytest.mark.parametrize("field", ["source", "evidence_checksum"])
def test_transition_rejects_unbound_decision_metadata(field):
    target = ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局")
    proposal = _proposal()
    decision_data = {
        "target_id": proposal.target_id,
        "source": proposal.source,
        "evidence_checksum": proposal.evidence_checksum,
        "approved": True,
    }
    decision_data[field] = "mismatched-value"
    decision = TargetDecision.model_validate(decision_data)

    with pytest.raises(ValueError, match="proposal"):
        transition_target(target, proposal, decision=decision, stage="feasibility_review")


def test_approved_target_change_before_build_increments_revision():
    target = ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局")
    proposal = _proposal()
    decision = TargetDecision.approve(proposal)

    transition = transition_target(target, proposal, decision=decision, stage="feasibility_review")

    assert transition.action == "revise_run"
    assert transition.target.target_id == "ai42.5Dic"
    assert transition.target.revision == 1
    assert transition.target_revision == 1
    assert transition.parent_run_id is None
    assert transition.target.approval_receipt is not None


@pytest.mark.parametrize(
    ("source", "evidence_checksum"),
    [("paper_evidence", "new-checksum"), ("new-source", "old-checksum")],
)
def test_approved_metadata_change_with_same_id_increments_revision(source, evidence_checksum):
    old_proposal = TargetProposal.from_paper_evidence(
        target_id="ai42.5Dic",
        display_name="ai42.5Dic",
        source="paper_evidence",
        evidence_checksum="old-checksum",
    )
    target = _approved_target(proposal=old_proposal, revision=4)
    proposal = TargetProposal.from_paper_evidence(
        target_id="ai42.5Dic",
        display_name="ai42.5Dic",
        source=source,
        evidence_checksum=evidence_checksum,
    )

    transition = transition_target(
        target,
        proposal,
        decision=TargetDecision.approve(proposal),
        stage="feasibility_review",
    )

    assert transition.action == "revise_run"
    assert transition.target_revision == 5
    assert transition.target.evidence_checksum == evidence_checksum
    assert transition.target.source == source


def test_target_change_after_build_requires_superseding_parent_run():
    target = ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局")
    proposal = _proposal()
    decision = TargetDecision.approve(proposal)

    with pytest.raises(ValueError, match="parent_run_id"):
        transition_target(target, proposal, decision=decision, stage="building")

    transition = transition_target(
        target,
        proposal,
        decision=decision,
        stage="building",
        parent_run_id="run-1",
    )
    assert transition.action == "supersede_run"
    assert transition.parent_run_id == "run-1"
    assert not transition.reuse_prior_artifacts


@pytest.mark.parametrize("stage", ["building", "verification", "version_archiving"])
def test_late_target_change_never_reuses_prior_run(stage):
    preview = transition_target("ai42.5Dic", stage=stage)

    assert preview.action == "supersede_preview"
    assert not preview.materialized


@pytest.mark.parametrize("parent_run_id", [None, "   "])
def test_materialized_supersede_requires_non_empty_parent_before_approval(parent_run_id):
    target = ResearchTarget.seed("rectilinear-soft-module", "直角多边形软模块布局")

    with pytest.raises(ValidationError, match="parent_run_id"):
        TargetTransition(
            action="supersede_run",
            target=target,
            target_revision=target.revision,
            parent_run_id=parent_run_id,
            approved=False,
        )


def test_run_config_loads_backward_compatible_unapproved_seed(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("topic: layout\ndata_root: %s\n" % tmp_path, encoding="utf-8")
    config = load_config(config_path)

    assert config.research_target.is_seed
    assert config.target_revision == 0
    assert config.target_evidence_checksum == "seed"
    with pytest.raises(ValueError, match="approved"):
        config.require_approved_target()


def test_runtime_run_config_requires_target_metadata_before_construction(tmp_path):
    with pytest.raises(ValidationError, match="research_target"):
        RunConfig.model_validate({"topic": "layout", "data_root": str(tmp_path)})


def test_manifest_requires_target_metadata_and_parent_cannot_equal_run():
    with pytest.raises(ValidationError):
        RunManifest.model_validate({"run_id": "run-1"})

    with pytest.raises(ValidationError, match="parent_run_id"):
        RunManifest.model_validate(_manifest_payload(parent_run_id="run-1"))


def test_unified_manifest_boundary_rejects_self_parent():
    with pytest.raises(ValidationError, match="parent_run_id"):
        validate_manifest_payload(
            _manifest_payload(parent_run_id="run-1", target_transition_action="supersede_run")
        )


def test_approved_manifest_requires_approved_target():
    with pytest.raises(ValidationError, match="approved"):
        RunManifest.model_validate(_manifest_payload(status="approved"))


def test_superseded_manifest_requires_parent_and_explicit_transition():
    with pytest.raises(ValidationError, match="parent_run_id"):
        RunManifest.model_validate(_manifest_payload(status="superseded"))
    with pytest.raises(ValidationError, match="supersede_run"):
        RunManifest.model_validate(_manifest_payload(parent_run_id="parent-run"))

    manifest = RunManifest.model_validate(
        _manifest_payload(parent_run_id="parent-run", target_transition_action="supersede_run")
    )
    assert manifest.parent_run_id == "parent-run"


def test_manifest_schema_requires_target_metadata_and_lineage_rules():
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "manifest.schema.json").read_text(encoding="utf-8"))
    valid = RunManifest.model_validate(_manifest_payload()).model_dump(mode="json", by_alias=True)

    jsonschema.validate(valid, schema)
    legacy = {key: value for key, value in valid.items() if key != "human_gates_enabled"}
    jsonschema.validate(legacy, schema)
    jsonschema.validate({**legacy, "human_gates_enabled": False}, schema)
    missing = {key: value for key, value in valid.items() if key != "research_target"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**valid, "status": "superseded"}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**valid, "parent_run_id": "parent-run"}, schema)
    self_parent = {
        **valid,
        "parent_run_id": valid["run_id"],
        "target_transition_action": "supersede_run",
    }
    jsonschema.validate(self_parent, schema)
    with pytest.raises(ValidationError, match="parent_run_id"):
        validate_manifest_payload(self_parent)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(
            {
                **valid,
                "status": "approved",
                "research_target": {**valid["research_target"], "approved": False},
            },
            schema,
        )


def test_run_schema_requires_normalized_target_metadata():
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "run.schema.json").read_text(encoding="utf-8"))
    valid = RunConfig.model_validate(
        {
            "topic": "layout",
            "data_root": "D:/runs",
            "research_target": ResearchTarget.seed("ai42.5Dic", "ai42.5Dic"),
        }
    ).model_dump(mode="json", by_alias=True)

    jsonschema.validate(valid, schema)
    missing = {key: value for key, value in valid.items() if key != "research_target"}
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing, schema)

    for field, value in (("target_revision", 99), ("target_evidence_checksum", "mismatch")):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**valid, field: value}, schema)


def test_run_schema_requires_approval_receipt_for_approved_target():
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "run.schema.json").read_text(encoding="utf-8"))
    config = RunConfig.model_validate(
        {
            "topic": "layout",
            "data_root": "D:/runs",
            "research_target": _approved_target(),
        }
    )
    valid = config.model_dump(mode="json", by_alias=True)
    jsonschema.validate(valid, schema)

    missing_receipt = {**valid, "research_target": {**valid["research_target"]}}
    missing_receipt["research_target"].pop("approval_receipt")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_receipt, schema)


@pytest.mark.parametrize(
    "payload",
    [{"papers": [{"path": "   "}]}, {"obsidian": {"vault": "   "}}],
)
def test_run_schema_rejects_whitespace_only_input_paths(payload):
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "run.schema.json").read_text(encoding="utf-8"))
    valid = RunConfig.model_validate(
        {
            "topic": "layout",
            "data_root": "D:/runs",
            "research_target": ResearchTarget.seed("ai42.5Dic", "ai42.5Dic"),
        }
    ).model_dump(mode="json", by_alias=True)

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**valid, **payload}, schema)


def test_manifest_schema_rejects_duplicated_target_metadata_fields():
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "manifest.schema.json").read_text(encoding="utf-8"))
    valid = RunManifest.model_validate(_manifest_payload()).model_dump(mode="json", by_alias=True)

    for field, value in (("target_revision", 99), ("target_evidence_checksum", "different")):
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.validate({**valid, field: value}, schema)


def test_manifest_schema_requires_approval_receipt_for_approved_target():
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "manifest.schema.json").read_text(encoding="utf-8"))
    manifest = RunManifest.model_validate(
        _manifest_payload(status="approved", research_target=_approved_target())
    )
    valid = manifest.model_dump(mode="json", by_alias=True)
    jsonschema.validate(valid, schema)

    missing_receipt = {**valid, "research_target": {**valid["research_target"]}}
    missing_receipt["research_target"].pop("approval_receipt")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(missing_receipt, schema)


@pytest.mark.parametrize("field", ["id", "display_name", "source", "evidence_checksum"])
def test_target_schema_rejects_whitespace_only_strings(field):
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "target.schema.json").read_text(encoding="utf-8"))
    target = {
        "id": "ai42.5Dic",
        "display_name": "ai42.5Dic",
        "revision": 0,
        "source": "paper_evidence",
        "evidence_checksum": "evidence-1",
        "approved": False,
    }

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**target, field: "   "}, schema)


@pytest.mark.parametrize("field", ["run_id", "source_commit", "config_checksum", "metric_schema_checksum", "executor", "model_provider"])
def test_manifest_rejects_whitespace_only_identifiers(field):
    payload = _manifest_payload()
    payload[field] = "   "

    with pytest.raises(ValidationError):
        RunManifest.model_validate(payload)


def test_target_schema_accepts_opaque_id_and_rejects_missing_evidence(tmp_path):
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "target.schema.json").read_text(encoding="utf-8"))
    target = _approved_target()

    jsonschema.validate(target.model_dump(by_alias=True), schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**target.model_dump(by_alias=True), "evidence_checksum": ""}, schema)


def test_target_schema_accepts_either_python_alias_but_not_both():
    schema = json.loads((Path(__file__).parents[1] / "schemas" / "target.schema.json").read_text(encoding="utf-8"))
    target = {
        "display_name": "ai42.5Dic",
        "revision": 0,
        "source": "user_seed",
        "evidence_checksum": "seed",
        "approved": False,
    }

    jsonschema.validate({**target, "id": "ai42.5Dic"}, schema)
    jsonschema.validate({**target, "target_id": "ai42.5Dic"}, schema)
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate({**target, "id": "ai42.5Dic", "target_id": "other"}, schema)


def test_manifest_serializes_target_metadata():
    manifest = RunManifest.model_validate(_manifest_payload())

    assert manifest.target_revision == 0
    assert manifest.research_target.target_id == "rectilinear-soft-module"
    assert manifest.model_dump(mode="json")["research_target"]["evidence_checksum"] == "seed"
