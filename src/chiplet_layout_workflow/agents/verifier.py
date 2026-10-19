from __future__ import annotations

import json
from collections.abc import Mapping

import yaml
from pydantic import ValidationError

from ..contracts import Decision, MetricCandidate
from .base import BaseAgent, RunContext


class VerifierAgent(BaseAgent):
    name = "verifier"
    role = "verifier"
    stage = "verification"
    required_outputs = (
        "evaluation_report.md",
        "metric_candidates.yaml",
        "approved_metrics.yaml",
        "comparison_table.csv",
        "verification_manifest.json",
        "results.json (placeholder only)",
    )

    def validate_inputs(self, context: RunContext) -> None:
        decision = context.artifact_path("feasibility_review", "decision.json")
        builder_result = context.artifact_path("building", "builder_result.json")
        candidates_path = context.artifact_path(
            "feasibility_review", "metric_candidates.yaml"
        )
        approved_metrics = context.artifact_path(
            "verification", "approved_metrics.yaml"
        )
        if not decision.is_file():
            raise ValueError("verifier requires a feasibility decision")
        try:
            decision_payload = json.loads(decision.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("verifier requires a valid feasibility decision") from exc
        if not isinstance(decision_payload, Mapping):
            raise ValueError("verifier requires a valid feasibility decision object")
        try:
            parsed_decision = Decision.model_validate(decision_payload)
        except ValidationError as exc:
            raise ValueError("verifier requires a valid feasibility decision") from exc
        if parsed_decision.decision != "accept":
            raise ValueError("verifier requires an approved feasibility decision")
        if not builder_result.is_file():
            raise ValueError("verifier requires an approved builder result")
        try:
            builder_payload = json.loads(builder_result.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError("verifier requires valid builder approval metadata") from exc
        if (
            not isinstance(builder_payload, Mapping)
            or builder_payload.get("approved") is not True
        ):
            raise ValueError("verifier requires an approved builder result")
        if not candidates_path.is_file():
            raise ValueError("verifier requires paper-backed metric candidates")
        if not approved_metrics.is_file():
            raise ValueError(
                "verifier requires approved_metrics.yaml with an approved metric"
            )
        try:
            candidates_payload = yaml.safe_load(
                candidates_path.read_text(encoding="utf-8")
            )
            approved_payload = yaml.safe_load(
                approved_metrics.read_text(encoding="utf-8")
            )
        except (OSError, UnicodeError, yaml.YAMLError) as exc:
            raise ValueError("verifier requires valid metric YAML files") from exc

        for label, payload in (
            ("paper-backed metric candidates", candidates_payload),
            ("approved metrics", approved_payload),
        ):
            if not isinstance(payload, Mapping) or not isinstance(
                payload.get("metrics"), list
            ):
                raise ValueError(
                    f"verifier requires {label} YAML mapping with a metrics list"
                )

        try:
            candidate_metrics = [
                MetricCandidate.model_validate(item)
                for item in candidates_payload["metrics"]
            ]
            approved_items = [
                MetricCandidate.model_validate(item)
                for item in approved_payload["metrics"]
            ]
        except ValidationError as exc:
            raise ValueError("verifier requires valid metric candidates") from exc

        candidate_definitions = {
            (item.name, item.direction, item.source, item.primary)
            for item in candidate_metrics
        }
        if not any(item.approved for item in approved_items):
            raise ValueError("approved_metrics.yaml must contain an approved metric")
        if any(
            (item.name, item.direction, item.source, item.primary)
            not in candidate_definitions
            for item in approved_items
        ):
            raise ValueError("every approved metric must come from paper candidates")
