"""Runtime validation boundaries for serialized workflow payloads.

Public callers should use these functions when accepting target, run, or
manifest data from files or other process boundaries. Each function first
checks the Draft 2020-12 structural contract and then applies the Pydantic
semantic contract, including relationships that JSON Schema cannot compare
dynamically.
"""

from __future__ import annotations

import json
from functools import lru_cache
from importlib import resources
from pathlib import Path
from typing import Any

import jsonschema
from pydantic import BaseModel

from .contracts import RunConfig, RunManifest, normalize_run_config_input
from .research_target import ResearchTarget


_SCHEMA_PACKAGE = "chiplet_layout_workflow.schema_assets"


class PayloadSchemaError(ValueError):
    """Raised when a payload violates its serialized JSON Schema contract."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


@lru_cache(maxsize=None)
def _schema_validator(schema_name: str) -> Any:
    schema_resource = resources.files(_SCHEMA_PACKAGE).joinpath(f"{schema_name}.schema.json")
    try:
        schema = json.loads(schema_resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Unable to load {schema_name} schema from package resources") from exc
    validator_cls = jsonschema.Draft202012Validator
    validator_cls.check_schema(schema)
    return validator_cls(schema)


def _validate_schema(schema_name: str, payload: Any) -> None:
    try:
        _schema_validator(schema_name).validate(payload)
    except jsonschema.ValidationError as exc:
        detail = exc
        while detail.context:
            detail = max(detail.context, key=lambda error: len(error.absolute_path))
        path = ".".join(str(part) for part in detail.absolute_path) or "payload"
        raise PayloadSchemaError(
            f"{schema_name} payload violates JSON Schema at {path}: {detail.message}"
        ) from exc


def validate_target_payload(payload: Any) -> ResearchTarget:
    """Validate a serialized target structurally and semantically."""

    normalized = _json_ready(payload)
    _validate_schema("target", normalized)
    return ResearchTarget.model_validate(normalized)


def validate_run_payload(payload: Any) -> RunConfig:
    """Validate a run/config payload at the runtime boundary.

    Backward-compatible seed defaults and string target shorthand are expanded
    before structural validation; user-provided values are not semantically
    validated until after the run schema has accepted their structure.
    """

    normalized = normalize_run_config_input(payload)
    _validate_schema("run", _json_ready(normalized))
    return RunConfig.model_validate(normalized)


def validate_manifest_payload(payload: Any) -> RunManifest:
    """Validate a serialized manifest, including dynamic lineage semantics."""

    normalized = _json_ready(payload)
    _validate_schema("manifest", normalized)
    return RunManifest.model_validate(normalized)
