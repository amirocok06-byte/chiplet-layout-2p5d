from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .research_target import ResearchTarget
from .target_policy import AI42_5DIC_DISPLAY_NAME, AI42_5DIC_TARGET_ID


def _path_string(value: object, *, field: str = "path") -> str:
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{field} must be a string or path-like value")
    path_value = os.fspath(value)
    if not isinstance(path_value, str):
        raise ValueError(f"{field} must be a text path")
    return path_value


def _normalized_path(value: Path | str) -> Path:
    """Normalize spelling without resolving or accessing the user's filesystem."""
    path_value = _path_string(value)
    return Path(os.path.normpath(os.path.expanduser(path_value)))


def _normalized_non_empty_path(value: Path | str, *, field: str) -> Path:
    path_value = _path_string(value, field=field)
    if not path_value.strip():
        raise ValueError(f"{field} must not be empty")
    return _normalized_path(path_value)


def ensure_within_root(path: Path | str, root: Path | str, *, field: str = "path") -> Path:
    """Return a lexical absolute path, rejecting traversal outside ``root``.

    This deliberately does not call ``Path.resolve``: validation must not require
    the path to exist or follow symlinks.
    """
    root_value = _path_string(root, field="root")
    root_abs = os.path.abspath(os.path.normpath(os.path.expanduser(root_value)))
    candidate = _path_string(path, field=field)
    if not os.path.isabs(candidate):
        candidate = os.path.join(root_abs, candidate)
    candidate_abs = os.path.abspath(os.path.normpath(os.path.expanduser(candidate)))
    try:
        common = os.path.commonpath((root_abs, candidate_abs))
    except ValueError as exc:
        raise ValueError(f"{field} must remain within configured root '{root_abs}'") from exc
    if common != root_abs:
        raise ValueError(f"{field} escapes configured root '{root_abs}'")
    return Path(candidate_abs)


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PaperInput(ContractModel):
    path: Path
    title: str | None = None

    @field_validator("path", mode="before")
    @classmethod
    def normalize_path(cls, value: Path | str) -> Path:
        return _normalized_non_empty_path(value, field="path")


class ObsidianInput(ContractModel):
    vault: Path
    include: list[str] = Field(default_factory=list)
    write_back: bool = False

    @field_validator("vault", mode="before")
    @classmethod
    def normalize_vault(cls, value: Path | str) -> Path:
        return _normalized_non_empty_path(value, field="vault")

    @model_validator(mode="after")
    def validate_include_paths(self) -> "ObsidianInput":
        for include in self.include:
            ensure_within_root(include, self.vault, field="obsidian.include")
        return self


class ZoteroInput(ContractModel):
    enabled: bool = False
    paths: list[Path] = Field(default_factory=list)

    _normalize_paths = field_validator("paths")(lambda values: [_normalized_path(value) for value in values])


class WebRetrievalConfig(ContractModel):
    enabled: bool = False
    allowed_domains: list[str] = Field(default_factory=list)


class ExecutionConfig(ContractModel):
    mode: Literal["local", "ssh"] = "local"
    command_allowlist: dict[str, list[str]] = Field(default_factory=dict)


class HumanGateConfig(ContractModel):
    enabled: bool = True


class UploadPolicy(ContractModel):
    default_policy: Literal["allow", "deny"] = "deny"
    allowed_paths: list[Path] = Field(default_factory=list)

    _normalize_paths = field_validator("allowed_paths")(
        lambda values: [_normalized_path(value) for value in values]
    )


def normalize_run_config_input(value: object) -> object:
    """Expand backward-compatible config input into a complete runtime config."""
    if not isinstance(value, dict):
        return value
    data = dict(value)
    if "research_target" not in data:
        data["research_target"] = {
            "id": AI42_5DIC_TARGET_ID,
            "display_name": AI42_5DIC_DISPLAY_NAME,
            "revision": 0,
            "source": "user_seed",
            "evidence_checksum": "seed",
            "approved": False,
        }
    else:
        target = data["research_target"]
        if isinstance(target, str):
            data["research_target"] = {
                "id": target,
                "display_name": target,
                "revision": 0,
                "source": "user_seed",
                "evidence_checksum": "seed",
                "approved": False,
            }
    return data


class RunConfig(ContractModel):
    topic: str = Field(min_length=1)
    data_root: Path
    papers: list[PaperInput] = Field(default_factory=list)
    obsidian: ObsidianInput | None = None
    zotero: ZoteroInput = Field(default_factory=ZoteroInput)
    web_retrieval: WebRetrievalConfig = Field(default_factory=WebRetrievalConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    human_gates: HumanGateConfig = Field(default_factory=HumanGateConfig)
    upload: UploadPolicy = Field(default_factory=UploadPolicy)
    research_target: ResearchTarget
    parent_run_id: str | None = None

    @classmethod
    def from_input(cls, value: object) -> "RunConfig":
        return cls.model_validate(normalize_run_config_input(value))

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, value: str) -> str:
        return _require_non_empty_identifier(value, "topic")

    @field_validator("data_root", mode="before")
    @classmethod
    def normalize_data_root(cls, value: Path | str) -> Path:
        if isinstance(value, str) and not value.strip():
            raise ValueError("data_root must not be empty")
        return _normalized_path(value)

    @model_validator(mode="after")
    def validate_rooted_paths(self) -> "RunConfig":
        self.upload.allowed_paths = [
            ensure_within_root(path, self.data_root, field="upload.allowed_paths")
            for path in self.upload.allowed_paths
        ]
        return self

    @field_validator("parent_run_id")
    @classmethod
    def validate_identifiers(cls, value: str | None, info) -> str | None:
        return None if value is None else _require_non_empty_identifier(value, info.field_name)

    @property
    def target_revision(self) -> int:
        return self.research_target.revision

    @property
    def target_evidence_checksum(self) -> str:
        return self.research_target.evidence_checksum

    def require_approved_target(self) -> ResearchTarget:
        return self.research_target.require_approved()


class Decision(ContractModel):
    decision: Literal["accept", "revise", "reject"]
    confidence: float = Field(ge=0, le=1)
    blocking_risks: list[str] = Field(default_factory=list)
    human_questions: list[str] = Field(default_factory=list)
    notes: str | None = None


class MetricCandidate(ContractModel):
    name: str = Field(min_length=1)
    direction: Literal["minimize", "maximize"]
    approved: bool
    source: str | None = None
    primary: bool = False


class RunManifest(ContractModel):
    run_id: str = Field(min_length=1)
    checksums: dict[str, str]
    source_commit: str = Field(min_length=1)
    config_checksum: str = Field(min_length=1)
    metric_schema_checksum: str = Field(min_length=1)
    result_checksums: dict[str, str] = Field(default_factory=dict)
    executor: str = Field(min_length=1)
    model_provider: str = Field(min_length=1)
    status: Literal["pending", "approved", "rejected", "superseded", "failed"]
    research_target: ResearchTarget
    human_gates_enabled: bool = True
    parent_run_id: str | None = None
    target_transition_action: Literal["no_change", "revise_run", "supersede_run"] | None = None

    @model_validator(mode="after")
    def validate_target_metadata(self) -> "RunManifest":
        if self.status == "approved":
            try:
                self.research_target.require_approved()
            except ValueError as exc:
                raise ValueError(
                    "approved manifest requires an approved research target with approval provenance"
                ) from exc
        if self.status == "superseded" and self.parent_run_id is None:
            raise ValueError("superseded manifest requires parent_run_id")
        if self.parent_run_id is not None and self.target_transition_action != "supersede_run":
            raise ValueError("parent_run_id requires target_transition_action='supersede_run'")
        if self.target_transition_action == "supersede_run" and self.parent_run_id is None:
            raise ValueError("supersede_run transition requires parent_run_id")
        if self.status == "superseded" and self.target_transition_action != "supersede_run":
            raise ValueError("superseded manifest requires target_transition_action='supersede_run'")
        if self.parent_run_id is not None:
            _require_non_empty_identifier(self.parent_run_id, "parent_run_id")
            if self.parent_run_id == self.run_id:
                raise ValueError("parent_run_id must differ from run_id")
        return self

    @property
    def target_revision(self) -> int:
        return self.research_target.revision

    @property
    def target_evidence_checksum(self) -> str:
        return self.research_target.evidence_checksum

    @field_validator("run_id", "source_commit", "config_checksum", "metric_schema_checksum", "executor", "model_provider")
    @classmethod
    def validate_identifiers(cls, value: str, info) -> str:
        return _require_non_empty_identifier(value, info.field_name)

    @field_validator("checksums", "result_checksums")
    @classmethod
    def validate_checksum_values(cls, value: dict[str, str]) -> dict[str, str]:
        return {key: _require_non_empty_identifier(checksum, f"checksum[{key}]") for key, checksum in value.items()}

    def require_approved_target(self) -> ResearchTarget:
        return self.research_target.require_approved()


class BuilderApprovalReceipt(ContractModel):
    """Auditable, human-gated authorization for the Retriever-to-Builder handoff."""

    receipt_version: Literal[1] = 1
    run_id: str = Field(min_length=1)
    target: ResearchTarget
    reader_manifest_checksum: str = Field(min_length=1)
    retriever_manifest_checksum: str = Field(min_length=1)
    decision_checksum: str = Field(min_length=1)
    approval_status: Literal["approved"]
    human_confirmation: Literal["explicit"]
    operator_note: str = Field(min_length=1)
    receipt_checksum: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_integrity(self) -> "BuilderApprovalReceipt":
        self.target.require_approved()
        for field_name in (
            "run_id",
            "reader_manifest_checksum",
            "retriever_manifest_checksum",
            "decision_checksum",
            "operator_note",
        ):
            _require_non_empty_identifier(getattr(self, field_name), field_name)
        expected_checksum = self.checksum_for_payload(self.model_dump(mode="json", by_alias=True))
        if self.receipt_checksum != expected_checksum:
            raise ValueError("builder approval receipt checksum does not match its payload")
        return self

    @staticmethod
    def checksum_for_payload(payload: dict[str, object]) -> str:
        unsigned = dict(payload)
        unsigned.pop("receipt_checksum", None)
        canonical = json.dumps(
            unsigned,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    @classmethod
    def issue(
        cls,
        *,
        run_id: str,
        target: ResearchTarget,
        reader_manifest_checksum: str,
        retriever_manifest_checksum: str,
        decision_checksum: str,
        gate_result: object,
        operator_note: str | None = None,
    ) -> "BuilderApprovalReceipt":
        from .gates.human_gate import GateResult

        if not isinstance(gate_result, GateResult):
            raise ValueError("builder approval receipt requires a HumanGate result")
        if (
            gate_result.enabled is not True
            or gate_result.decision != "approve"
            or gate_result.status != "approved"
        ):
            raise ValueError("builder approval receipt requires explicit human approval")
        note = gate_result.operator_note if operator_note is None else operator_note
        if not isinstance(note, str) or not note.strip():
            raise ValueError("builder approval receipt requires an operator note")
        payload = {
            "receipt_version": 1,
            "run_id": run_id,
            "target": target.model_dump(mode="json", by_alias=True),
            "reader_manifest_checksum": reader_manifest_checksum,
            "retriever_manifest_checksum": retriever_manifest_checksum,
            "decision_checksum": decision_checksum,
            "approval_status": "approved",
            "human_confirmation": "explicit",
            "operator_note": note,
        }
        payload["receipt_checksum"] = cls.checksum_for_payload(payload)
        return cls.model_validate(payload)


def _require_non_empty_identifier(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value
