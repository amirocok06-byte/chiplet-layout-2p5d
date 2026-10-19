from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, TYPE_CHECKING

from pydantic import ValidationError

from .artifacts.checksums import sha256_file
from .contracts import BuilderApprovalReceipt, Decision

if TYPE_CHECKING:
    from .agents.base import RunContext
    from .gates.human_gate import GateResult


BUILDER_APPROVAL_RECEIPT_FILENAME = "builder_approval_receipt.json"
_READER_SUBSTANTIVE_OUTPUTS = (
    "paper_understanding.md",
    "core_method.md",
    "candidate_plan.md",
    "transferable_ideas.md",
    "limitations.md",
    "citations.json",
    "target_assessment.json",
)
_READER_METADATA_OUTPUTS = ("agent_status.json", "reader_output_manifest.json")
_RETRIEVER_REQUIRED_OUTPUTS = {
    "feasibility_review.md",
    "revised_plan.md",
    "evidence_table.md",
    "metric_candidates.yaml",
    "decision.json",
    "agent_status.json",
    "retriever_output_manifest.json",
}
_RETRIEVER_ALLOWED_OUTPUTS = _RETRIEVER_REQUIRED_OUTPUTS | {"target_change.json"}


class InvalidBuilderHandoff(ValueError):
    """Raised when the Reader/Retriever handoff is not safe to approve or build."""


def _read_json_mapping(path: Path, description: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise InvalidBuilderHandoff(f"builder requires valid {description} JSON") from exc
    if not isinstance(payload, dict):
        raise InvalidBuilderHandoff(f"builder requires {description} to be a JSON mapping")
    return payload


def _require_files(root: Path, names: tuple[str, ...] | set[str], description: str) -> None:
    missing = [name for name in names if not (root / name).is_file()]
    if missing:
        raise InvalidBuilderHandoff(
            f"builder requires complete {description}: {', '.join(sorted(missing))}"
        )


def _validate_reader_handoff(context: RunContext) -> None:
    root = context.run.paper_understanding
    _require_files(
        root,
        set(_READER_SUBSTANTIVE_OUTPUTS + _READER_METADATA_OUTPUTS),
        "reader artifacts",
    )
    manifest = _read_json_mapping(
        root / "reader_output_manifest.json", "reader output manifest"
    )
    if (
        manifest.get("agent") != "reader"
        or manifest.get("complete") is not True
        or manifest.get("metadata_outputs") != list(_READER_METADATA_OUTPUTS)
        or manifest.get("response_model") != "ReaderOutputEnvelope"
        or manifest.get("stage") != "reading"
        or manifest.get("status") != "completed"
        or manifest.get("target_id") != context.current_target.target_id
    ):
        raise InvalidBuilderHandoff("builder requires a complete reader output manifest")
    substantive_outputs = manifest.get("substantive_outputs")
    if (
        not isinstance(substantive_outputs, list)
        or set(substantive_outputs) != set(_READER_SUBSTANTIVE_OUTPUTS)
        or len(substantive_outputs) != len(_READER_SUBSTANTIVE_OUTPUTS)
    ):
        raise InvalidBuilderHandoff("builder requires reader manifest substantive outputs")

    status = _read_json_mapping(root / "agent_status.json", "reader agent status")
    if (
        status.get("agent") != "reader"
        or status.get("execution") != "structured_provider_response"
        or status.get("stage") != "reading"
        or status.get("status") != "completed"
        or status.get("target")
        != context.current_target.model_dump(mode="json", by_alias=True)
    ):
        raise InvalidBuilderHandoff(
            "builder requires reader agent status for the current target"
        )


def _validate_retriever_handoff(context: RunContext) -> tuple[Path, dict[str, Any]]:
    root = context.run.feasibility
    _require_files(root, _RETRIEVER_REQUIRED_OUTPUTS, "retriever artifacts")
    manifest_path = root / "retriever_output_manifest.json"
    manifest = _read_json_mapping(manifest_path, "retriever output manifest")
    target_payload = context.current_target.model_dump(mode="json", by_alias=True)
    if (
        manifest.get("agent") != "retriever"
        or manifest.get("complete") is not True
        or manifest.get("run_id") != context.run.run_id
        or manifest.get("stage") != "feasibility_review"
        or manifest.get("status") != "completed"
        or manifest.get("response_model") != "RetrieverOutputEnvelope"
        or manifest.get("target") != target_payload
        or manifest.get("target_id") != context.current_target.target_id
        or manifest.get("target_revision") != context.current_target.revision
        or manifest.get("target_evidence_checksum") != context.current_target.evidence_checksum
    ):
        raise InvalidBuilderHandoff(
            "builder requires a retriever manifest for the current run and target"
        )

    outputs = manifest.get("outputs")
    output_checksums = manifest.get("output_checksums")
    if not isinstance(outputs, list) or not isinstance(output_checksums, dict):
        raise InvalidBuilderHandoff("builder requires retriever output checksums")
    output_set = set(outputs)
    if (
        len(outputs) != len(output_set)
        or not _RETRIEVER_REQUIRED_OUTPUTS.issubset(output_set)
        or not output_set.issubset(_RETRIEVER_ALLOWED_OUTPUTS)
    ):
        raise InvalidBuilderHandoff("builder requires the complete retriever output set")
    checksum_names = output_set - {"retriever_output_manifest.json"}
    if set(output_checksums) != checksum_names:
        raise InvalidBuilderHandoff(
            "builder requires retriever manifest checksums to match outputs"
        )
    for name in checksum_names:
        path = root / name
        if not path.is_file() or sha256_file(path) != output_checksums[name]:
            raise InvalidBuilderHandoff(
                f"builder retriever output checksum mismatch: {name}"
            )
    return manifest_path, manifest


def collect_builder_handoff(context: RunContext) -> dict[str, str]:
    """Validate both upstream manifests and return the exact bytes they bind."""

    try:
        context.current_target.require_approved()
        context.run.research_target.require_approved()
    except ValueError as exc:
        raise InvalidBuilderHandoff("builder requires an approved current run target") from exc
    current_target = context.current_target.model_dump(mode="json", by_alias=True)
    run_target = context.run.research_target.model_dump(mode="json", by_alias=True)
    if current_target != run_target:
        raise InvalidBuilderHandoff("builder current target does not match the run target")

    _validate_reader_handoff(context)
    retriever_manifest_path, _ = _validate_retriever_handoff(context)
    decision_path = context.run.feasibility / "decision.json"
    try:
        decision = Decision.model_validate(
            _read_json_mapping(decision_path, "feasibility decision")
        )
    except (ValidationError, InvalidBuilderHandoff) as exc:
        raise InvalidBuilderHandoff("builder requires a valid feasibility decision") from exc
    if decision.decision != "accept":
        raise InvalidBuilderHandoff("builder requires an accepted feasibility decision")

    reader_manifest_path = context.run.paper_understanding / "reader_output_manifest.json"
    return {
        "reader_manifest_checksum": sha256_file(reader_manifest_path),
        "retriever_manifest_checksum": sha256_file(retriever_manifest_path),
        "decision_checksum": sha256_file(decision_path),
    }


def issue_builder_approval_receipt(
    context: RunContext,
    gate_result: GateResult,
    *,
    operator_note: str | None = None,
) -> BuilderApprovalReceipt:
    """Create a receipt only from an explicit enabled ``HumanGate`` result."""

    checksums = collect_builder_handoff(context)
    return BuilderApprovalReceipt.issue(
        run_id=context.run.run_id,
        target=context.current_target,
        gate_result=gate_result,
        operator_note=operator_note,
        **checksums,
    )


def write_builder_approval_receipt(
    context: RunContext,
    gate_result: GateResult,
    *,
    operator_note: str | None = None,
) -> Path:
    receipt = issue_builder_approval_receipt(
        context,
        gate_result,
        operator_note=operator_note,
    )
    target = context.run.feasibility / BUILDER_APPROVAL_RECEIPT_FILENAME
    context.run.assert_inside(target)
    payload = json.dumps(
        receipt.model_dump(mode="json", by_alias=True),
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=".builder-approval-receipt-", dir=str(target.parent)
    )
    os.close(file_descriptor)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(payload, encoding="utf-8")
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def validate_builder_approval_receipt(
    context: RunContext,
) -> BuilderApprovalReceipt:
    checksums = collect_builder_handoff(context)
    path = context.run.feasibility / BUILDER_APPROVAL_RECEIPT_FILENAME
    context.run.assert_inside(path)
    try:
        receipt = BuilderApprovalReceipt.model_validate(
            _read_json_mapping(path, "builder approval receipt")
        )
    except (ValidationError, InvalidBuilderHandoff) as exc:
        raise InvalidBuilderHandoff("builder requires a valid approval receipt") from exc
    if (
        receipt.run_id != context.run.run_id
        or receipt.target.model_dump(mode="json", by_alias=True)
        != context.current_target.model_dump(mode="json", by_alias=True)
        or receipt.reader_manifest_checksum != checksums["reader_manifest_checksum"]
        or receipt.retriever_manifest_checksum != checksums["retriever_manifest_checksum"]
        or receipt.decision_checksum != checksums["decision_checksum"]
    ):
        raise InvalidBuilderHandoff(
            "builder approval receipt is not bound to the current handoff"
        )
    return receipt


__all__ = [
    "BUILDER_APPROVAL_RECEIPT_FILENAME",
    "InvalidBuilderHandoff",
    "collect_builder_handoff",
    "issue_builder_approval_receipt",
    "validate_builder_approval_receipt",
    "write_builder_approval_receipt",
]
