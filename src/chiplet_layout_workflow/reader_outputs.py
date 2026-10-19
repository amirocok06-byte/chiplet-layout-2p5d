from __future__ import annotations

from copy import deepcopy
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .research_target import TargetProposal


class _ReaderModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class ReaderCitation(_ReaderModel):
    source: str = Field(min_length=1)
    source_kind: Literal["reference_paper", "ai_placement_note"]
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    locator: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    title: str | None = Field(default=None, min_length=1)

    @field_validator("source", "locator", "claim")
    @classmethod
    def validate_required_text(cls, value: str, info) -> str:
        return _require_non_empty_text(value, info.field_name)

    @field_validator("title")
    @classmethod
    def validate_optional_title(cls, value: str | None) -> str | None:
        return None if value is None else _require_non_empty_text(value, "title")


class ReaderEvidence(_ReaderModel):
    source: str = Field(min_length=1)
    claim: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @field_validator("source", "claim")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _require_non_empty_text(value, info.field_name)


class ReaderTargetAssessment(_ReaderModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "allOf": [
                {
                    "if": {
                        "properties": {"status": {"const": "needs_change"}},
                        "required": ["status"],
                    },
                    "then": {
                        "required": ["proposal"],
                        "properties": {"proposal": {"type": "object"}},
                    },
                },
                {
                    "if": {
                        "properties": {"status": {"const": "confirmed"}},
                        "required": ["status"],
                    },
                    "then": {"properties": {"proposal": {"type": "null"}}},
                },
                {
                    "if": {
                        "properties": {"proposal": {"type": "object"}},
                        "required": ["proposal"],
                    },
                    "then": {
                        "properties": {
                            "proposal": {
                                "properties": {
                                    "approved": {"const": False},
                                    "approval_state": {"const": "pending"},
                                    "requires_human_approval": {"const": True},
                                }
                            }
                        }
                    },
                },
            ]
        },
    )
    current_target_id: str = Field(min_length=1)
    status: Literal["confirmed", "uncertain", "needs_change"]
    rationale: str = Field(min_length=1)
    evidence: list[ReaderEvidence] = Field(min_length=1)
    human_questions: list[str]
    proposal: TargetProposal | None = None

    @field_validator("current_target_id", "rationale")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _require_non_empty_text(value, info.field_name)

    @field_validator("human_questions")
    @classmethod
    def validate_human_questions(cls, value: list[str]) -> list[str]:
        return [
            _require_non_empty_text(question, "human_questions item")
            for question in value
        ]

    @model_validator(mode="after")
    def validate_proposal_is_pending(self) -> "ReaderTargetAssessment":
        if self.status == "needs_change" and self.proposal is None:
            raise ValueError("reader assessments needing a change must include a proposal")
        if self.status == "confirmed" and self.proposal is not None:
            raise ValueError("confirmed reader assessments must not include a proposal")
        if self.proposal is not None and (
            self.proposal.approved is not False
            or self.proposal.approval_state != "pending"
            or self.proposal.requires_human_approval is not True
        ):
            raise ValueError(
                "reader proposals must be unapproved, pending, and require human approval"
            )
        return self


class ReaderOutputEnvelope(_ReaderModel):
    paper_understanding: str = Field(min_length=1)
    core_method: str = Field(min_length=1)
    candidate_plan: str = Field(min_length=1)
    transferable_ideas: str = Field(min_length=1)
    limitations: str = Field(min_length=1)
    citations: list[ReaderCitation] = Field(min_length=1)
    target_assessment: ReaderTargetAssessment

    @field_validator(
        "paper_understanding",
        "core_method",
        "candidate_plan",
        "transferable_ideas",
        "limitations",
    )
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _require_non_empty_text(value, info.field_name)


def reader_response_schema() -> dict[str, Any]:
    schema = ReaderOutputEnvelope.model_json_schema()

    def add_non_empty_text_patterns(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "string" and "pattern" not in node:
                node["pattern"] = r"\S"
            for value in node.values():
                add_non_empty_text_patterns(value)
        elif isinstance(node, list):
            for item in node:
                add_non_empty_text_patterns(item)

    add_non_empty_text_patterns(schema)
    return schema


_UNSUPPORTED_PROVIDER_SCHEMA_KEYS = frozenset(
    {
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
)


def _to_provider_schema(node: Any) -> Any:
    if isinstance(node, list):
        return [_to_provider_schema(item) for item in node]
    if not isinstance(node, dict):
        return node

    transformed: dict[str, Any] = {}
    for key, value in node.items():
        if key in _UNSUPPORTED_PROVIDER_SCHEMA_KEYS:
            continue
        if key in {"properties", "$defs"} and isinstance(value, dict):
            transformed[key] = {
                child_key: _to_provider_schema(child_value)
                for child_key, child_value in value.items()
            }
        else:
            transformed[key] = _to_provider_schema(value)

    if transformed.get("type") == "object":
        properties = transformed.setdefault("properties", {})
        transformed["required"] = list(properties)
        transformed["additionalProperties"] = False

    return transformed


def reader_provider_response_schema() -> dict[str, Any]:
    return _to_provider_schema(deepcopy(reader_response_schema()))


__all__ = [
    "ReaderCitation",
    "ReaderEvidence",
    "ReaderTargetAssessment",
    "ReaderOutputEnvelope",
    "reader_response_schema",
    "reader_provider_response_schema",
]
