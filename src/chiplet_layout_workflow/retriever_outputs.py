from __future__ import annotations

from copy import deepcopy
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .contracts import Decision, MetricCandidate
from .research_target import TargetProposal


class _RetrieverModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_non_empty_text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


class RetrieverOutputEnvelope(_RetrieverModel):
    """The complete, provider-neutral response for one Retriever run."""

    feasibility_review: str = Field(min_length=1)
    revised_plan: str = Field(min_length=1)
    evidence_table: str = Field(min_length=1)
    metric_candidates: list[MetricCandidate] = Field(min_length=1)
    decision: Decision
    target_change: TargetProposal | None = None

    @field_validator("feasibility_review", "revised_plan", "evidence_table")
    @classmethod
    def validate_text(cls, value: str, info) -> str:
        return _require_non_empty_text(value, info.field_name)

    @model_validator(mode="after")
    def validate_unapproved_handoff(self) -> "RetrieverOutputEnvelope":
        if any(candidate.approved for candidate in self.metric_candidates):
            raise ValueError("Retriever metric candidates must remain unapproved")
        if self.target_change is not None and (
            self.target_change.approved is not False
            or self.target_change.approval_state != "pending"
            or self.target_change.requires_human_approval is not True
        ):
            raise ValueError(
                "Retriever target changes must be unapproved, pending, and require human approval"
            )
        return self


def retriever_response_schema() -> dict[str, Any]:
    schema = RetrieverOutputEnvelope.model_json_schema()

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


def retriever_provider_response_schema() -> dict[str, Any]:
    """Return a strict schema accepted by structured LLM providers."""

    return _to_provider_schema(deepcopy(retriever_response_schema()))


__all__ = [
    "RetrieverOutputEnvelope",
    "retriever_provider_response_schema",
    "retriever_response_schema",
]
