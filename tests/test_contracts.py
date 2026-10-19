import pytest
from pydantic import ValidationError

from chiplet_layout_workflow.config import ConfigError, load_config
from chiplet_layout_workflow.contracts import (
    Decision,
    MetricCandidate,
    ObsidianInput,
    PaperInput,
    RunConfig,
    RunManifest,
)
from chiplet_layout_workflow.research_target import ResearchTarget


def test_builder_approval_receipt_contract_is_explicit_and_auditable():
    from chiplet_layout_workflow import contracts

    receipt_type = getattr(contracts, "BuilderApprovalReceipt", None)

    assert receipt_type is not None


def test_run_config_requires_topic_and_data_root():
    config = RunConfig.model_validate(
        {
            "topic": "layout",
            "data_root": "D:/runs",
            "research_target": {
                "id": "ai42.5Dic",
                "display_name": "ai42.5Dic",
                "revision": 0,
                "source": "user_seed",
                "evidence_checksum": "seed",
                "approved": False,
            },
        }
    )

    assert config.topic == "layout"


def test_decision_rejects_unknown_values():
    with pytest.raises(ValidationError):
        Decision.model_validate({"decision": "maybe", "confidence": 0.5})


def test_run_config_rejects_upload_path_outside_data_root(tmp_path):
    with pytest.raises(ValidationError, match="upload.allowed_paths"):
        RunConfig.from_input(
            {
                "topic": "layout",
                "data_root": str(tmp_path / "runs"),
                "upload": {"allowed_paths": [str(tmp_path / "runs" / ".." / "outside")]},
            }
        )


def test_run_config_accepts_upload_child_path(tmp_path):
    data_root = tmp_path / "runs"
    config = RunConfig.from_input(
        {
            "topic": "layout",
            "data_root": str(data_root),
            "upload": {"allowed_paths": ["results"]},
        }
    )

    assert config.upload.allowed_paths == [data_root / "results"]


@pytest.mark.parametrize(
    "payload",
    [
        {"topic": "", "data_root": "D:/runs"},
        {"topic": "layout", "data_root": ""},
        {"topic": "layout", "data_root": "   "},
    ],
)
def test_run_config_rejects_empty_schema_required_strings(payload):
    with pytest.raises(ValidationError):
        RunConfig.from_input(payload)


def test_metric_candidate_rejects_empty_name():
    with pytest.raises(ValidationError):
        MetricCandidate.model_validate({"name": "", "direction": "minimize", "approved": False})


@pytest.mark.parametrize("blank", ["", "   "])
@pytest.mark.parametrize("model", [PaperInput, ObsidianInput])
def test_path_contracts_reject_empty_paths(model, blank):
    payload = {"path": blank} if model is PaperInput else {"vault": blank}

    with pytest.raises(ValidationError):
        model.model_validate(payload)


def test_path_contracts_accept_valid_paths(tmp_path):
    paper = PaperInput.model_validate({"path": tmp_path / "paper.md"})
    vault = ObsidianInput.model_validate({"vault": tmp_path / "vault", "include": ["notes.md"]})

    assert paper.path == tmp_path / "paper.md"
    assert vault.vault == tmp_path / "vault"
    assert vault.include == ["notes.md"]


@pytest.mark.parametrize(
    ("yaml_text", "field"),
    [
        ("topic: layout\ndata_root: 123\n", "data_root"),
        ("topic: layout\ndata_root: null\n", "data_root"),
        ("topic: layout\ndata_root: D:/runs\nobsidian:\n  vault: 123\n", "obsidian.vault"),
    ],
)
def test_load_config_rejects_malformed_path_scalars(tmp_path, yaml_text, field):
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text(yaml_text, encoding="utf-8")

    with pytest.raises(ConfigError, match=field):
        load_config(config_path)


@pytest.mark.parametrize(
    "field",
    ["run_id", "source_commit", "config_checksum", "metric_schema_checksum", "executor", "model_provider"],
)
def test_run_manifest_rejects_empty_identifiers(field):
    payload = {
        "run_id": "run-1",
        "checksums": {"input": "abc"},
        "source_commit": "commit-1",
        "config_checksum": "config-1",
        "metric_schema_checksum": "metric-1",
        "executor": "local",
        "model_provider": "mock",
        "status": "pending",
    }
    payload[field] = ""

    with pytest.raises(ValidationError):
        RunManifest.model_validate(payload)


def test_run_manifest_gate_metadata_defaults_enabled_and_accepts_disabled():
    payload = {
        "run_id": "run-1",
        "checksums": {"input": "abc"},
        "source_commit": "commit-1",
        "config_checksum": "config-1",
        "metric_schema_checksum": "metric-1",
        "executor": "local",
        "model_provider": "mock",
        "status": "pending",
        "research_target": ResearchTarget.seed("rectilinear-soft-module", "layout"),
    }

    legacy = RunManifest.model_validate(payload)
    disabled = RunManifest.model_validate({**payload, "human_gates_enabled": False})

    assert legacy.human_gates_enabled is True
    assert disabled.human_gates_enabled is False
