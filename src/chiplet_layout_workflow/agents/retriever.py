from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

import yaml
from pydantic import ValidationError

from ..artifacts.checksums import sha256_file
from ..contracts import Decision
from ..local_materials import MaterialManifest, read_material_manifest
from ..reader_outputs import ReaderOutputEnvelope
from ..research_target import ResearchTarget, TargetProposal
from ..retriever_outputs import RetrieverOutputEnvelope, retriever_provider_response_schema
from .base import AgentProtocolError, AgentResult, BaseAgent, RunContext


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
_READER_AGENT_REQUIRED_OUTPUTS = (
    "paper_understanding.md",
    "core_method.md",
    "transferable_ideas.md",
    "limitations.md",
    "citations.json",
    "candidate_plan.md",
    "target_assessment.json",
)
_READER_RESPONSE_MODEL = "ReaderOutputEnvelope"
_READER_EXECUTION = "structured_provider_response"
_READER_ARTIFACT_OUTPUTS = _READER_SUBSTANTIVE_OUTPUTS + _READER_METADATA_OUTPUTS


class RetrieverAgent(BaseAgent):
    name = "retriever"
    role = "retriever"
    stage = "feasibility_review"
    required_outputs = (
        "feasibility_review.md",
        "revised_plan.md",
        "evidence_table.md",
        "metric_candidates.yaml",
        "decision.json",
        "target_change.json (when proposed)",
        "agent_status.json",
        "retriever_output_manifest.json",
    )

    _SUBSTANTIVE_OUTPUTS = (
        "feasibility_review.md",
        "revised_plan.md",
        "evidence_table.md",
        "metric_candidates.yaml",
        "decision.json",
    )
    _METADATA_OUTPUTS = ("agent_status.json", "retriever_output_manifest.json")

    @staticmethod
    def validate_decision(payload: Any) -> Decision:
        return Decision.model_validate(payload)

    @staticmethod
    def validate_target_change(
        payload: Any, current_target: ResearchTarget
    ) -> TargetProposal:
        proposal = TargetProposal.model_validate(payload)
        if proposal.target_id == current_target.target_id:
            raise ValueError("target_change must differ from the current target")
        return proposal

    def validate_inputs(self, context: RunContext) -> MaterialManifest:
        if context.stage is not None:
            context = RunContext(
                run=context.run,
                current_target=context.current_target,
                provider=context.provider,
                prompt_renderer=context.prompt_renderer,
                input_paths=context.input_paths,
                source_worktree=context.source_worktree,
                git_adapter=context.git_adapter,
                benchmark_executor=context.benchmark_executor,
                metadata=context.metadata,
            )
        missing = [
            name
            for name in _READER_ARTIFACT_OUTPUTS
            if not context.artifact_path("reading", name).is_file()
        ]
        if missing:
            raise ValueError(
                "retriever requires approved reader artifacts: " + ", ".join(missing)
            )

        manifest = self._read_json_mapping(
            context, "reader_output_manifest.json", "reader output manifest"
        )
        self._validate_reader_manifest(manifest, context.current_target.target_id)

        agent_status = self._read_json_mapping(
            context, "agent_status.json", "reader agent status"
        )
        self._validate_reader_agent_status(agent_status, context)

        envelope = self._read_reader_output(context)
        self._validate_reader_target_assessment(envelope, context)
        return self._read_material_manifest(context)

    @staticmethod
    def _read_json_mapping(
        context: RunContext, name: str, description: str
    ) -> dict[str, Any]:
        path = context.artifact_path("reading", name)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError(f"retriever requires valid {description} JSON") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"retriever requires {description} to be a JSON mapping")
        return payload

    @staticmethod
    def _validate_reader_manifest(
        payload: dict[str, Any], current_target_id: str
    ) -> None:
        if (
            payload.get("agent") != "reader"
            or payload.get("complete") is not True
            or payload.get("metadata_outputs") != list(_READER_METADATA_OUTPUTS)
            or payload.get("response_model") != _READER_RESPONSE_MODEL
            or payload.get("stage") != "reading"
            or payload.get("status") != "completed"
            or payload.get("target_id") != current_target_id
        ):
            raise ValueError(
                "retriever requires a complete reader output manifest for the reading stage"
            )

        substantive_outputs = payload.get("substantive_outputs")
        if (
            not isinstance(substantive_outputs, list)
            or len(substantive_outputs) != len(_READER_SUBSTANTIVE_OUTPUTS)
            or any(not isinstance(name, str) for name in substantive_outputs)
            or set(substantive_outputs) != set(_READER_SUBSTANTIVE_OUTPUTS)
        ):
            raise ValueError(
                "retriever requires manifest substantive_outputs to exactly match reader outputs"
            )

    @staticmethod
    def _validate_reader_agent_status(
        payload: dict[str, Any], context: RunContext
    ) -> None:
        if (
            payload.get("agent") != "reader"
            or payload.get("execution") != _READER_EXECUTION
            or payload.get("required_outputs") != list(_READER_AGENT_REQUIRED_OUTPUTS)
            or payload.get("stage") != "reading"
            or payload.get("status") != "completed"
            or payload.get("target")
            != context.current_target.model_dump(mode="json", by_alias=True)
        ):
            raise ValueError(
                "retriever requires completed reader agent status for the reading stage"
            )

    @staticmethod
    def _read_reader_output(context: RunContext) -> ReaderOutputEnvelope:
        try:
            root = context.artifact_root("reading")
            payload = {
                "paper_understanding": (root / "paper_understanding.md").read_text(
                    encoding="utf-8"
                ),
                "core_method": (root / "core_method.md").read_text(encoding="utf-8"),
                "candidate_plan": (root / "candidate_plan.md").read_text(
                    encoding="utf-8"
                ),
                "transferable_ideas": (root / "transferable_ideas.md").read_text(
                    encoding="utf-8"
                ),
                "limitations": (root / "limitations.md").read_text(encoding="utf-8"),
                "citations": json.loads(
                    (root / "citations.json").read_text(encoding="utf-8")
                ),
                "target_assessment": json.loads(
                    (root / "target_assessment.json").read_text(encoding="utf-8")
                ),
            }
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as exc:
            raise ValueError("retriever requires a valid reader output envelope") from exc

        try:
            return ReaderOutputEnvelope.model_validate(payload)
        except (TypeError, ValidationError) as exc:
            raise ValueError("retriever requires a valid reader output envelope") from exc

    @staticmethod
    def _validate_reader_target_assessment(
        envelope: ReaderOutputEnvelope, context: RunContext
    ) -> None:
        assessment = envelope.target_assessment
        current_target_id = context.current_target.target_id
        if assessment.current_target_id != current_target_id:
            raise ValueError("reader target assessment does not match the current target")
        if assessment.proposal is not None and assessment.proposal.target_id == current_target_id:
            raise ValueError("reader target proposal must differ from the current target")

    @staticmethod
    def _read_material_manifest(context: RunContext) -> MaterialManifest:
        manifest_path = context.run.inputs / "material_manifest.json"
        try:
            context._validate_explicit_input_path(manifest_path)
            manifest = read_material_manifest(manifest_path)
        except Exception as exc:
            raise ValueError("retriever requires a valid staged material manifest") from exc

        for material in manifest.materials:
            relative = PurePosixPath(material.staged_relative_path)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or "\\" in material.staged_relative_path
            ):
                raise ValueError("retriever material manifest contains an unsafe staged path")
            staged_path = context.run.inputs.joinpath(*relative.parts)
            try:
                context._validate_explicit_input_path(staged_path)
            except Exception as exc:
                raise ValueError("retriever material manifest points outside run.inputs") from exc
            if sha256_file(staged_path) != material.staged_sha256:
                raise ValueError("retriever staged material checksum does not match manifest")
        return manifest

    def run(self, context: RunContext) -> AgentResult:
        context = self._stage_context(context)
        material_manifest = self.validate_inputs(context)
        evidence = self._staged_evidence(context, material_manifest)
        prompt = context.render_role_prompt(self.role, stage=self.stage)
        response = self.provider.generate(
            prompt,
            prompt,
            evidence,
            response_schema=retriever_provider_response_schema(),
        )
        envelope = self._parse_response(response)
        if envelope.target_change is not None:
            try:
                self.validate_target_change(envelope.target_change, context.current_target)
            except ValueError as exc:
                raise AgentProtocolError(str(exc)) from exc
        return self._publish(context, envelope, material_manifest, response)

    @staticmethod
    def _staged_evidence(
        context: RunContext, manifest: MaterialManifest
    ) -> list[Path]:
        paths = [context.run.inputs / "material_manifest.json"]
        paths.extend(
            context.run.inputs
            / Path(*PurePosixPath(material.staged_relative_path).parts)
            for material in manifest.materials
        )
        return sorted(paths)

    def _stage_context(self, context: RunContext) -> RunContext:
        if context.stage is not None and context.stage != self.stage:
            context.for_stage(self.stage)
        if context.provider is not self.provider or context.prompt_renderer is not self.prompt_renderer:
            context = RunContext(
                run=context.run,
                current_target=context.current_target,
                provider=self.provider,
                prompt_renderer=self.prompt_renderer,
                input_paths=context.input_paths,
                source_worktree=context.source_worktree,
                git_adapter=context.git_adapter,
                benchmark_executor=context.benchmark_executor,
                stage=context.stage,
                metadata=context.metadata,
            )
        return context.for_stage(self.stage)

    @staticmethod
    def _parse_response(response: Any) -> RetrieverOutputEnvelope:
        try:
            payload = json.loads(response.text)
        except (AttributeError, TypeError, json.JSONDecodeError) as exc:
            raise AgentProtocolError("retriever provider returned invalid JSON") from exc
        try:
            return RetrieverOutputEnvelope.model_validate(payload)
        except (TypeError, ValidationError) as exc:
            raise AgentProtocolError(
                "retriever provider response does not match the retriever output schema"
            ) from exc

    def _publish(
        self,
        context: RunContext,
        envelope: RetrieverOutputEnvelope,
        material_manifest: MaterialManifest,
        response: Any,
    ) -> AgentResult:
        root = context.artifact_root(self.stage)
        output_names = list(self._SUBSTANTIVE_OUTPUTS + self._METADATA_OUTPUTS)
        if envelope.target_change is not None:
            output_names.insert(5, "target_change.json")
        managed_names = set(self._SUBSTANTIVE_OUTPUTS + self._METADATA_OUTPUTS + ("target_change.json",))
        final_paths = {name: context.artifact_path(self.stage, name) for name in managed_names}
        published_paths: list[Path] = []
        backup_paths: dict[Path, Path] = {}
        temporary_root: Path | None = None

        try:
            temporary_root = Path(tempfile.mkdtemp(prefix=".retriever-", dir=str(root)))
            context.run.assert_inside(temporary_root)
            temporary_paths = {name: temporary_root / name for name in output_names}
            backup_root = temporary_root / "backup"
            backup_root.mkdir()

            serialized = envelope.model_dump(mode="json", by_alias=True)
            text_outputs = {
                "feasibility_review.md": envelope.feasibility_review,
                "revised_plan.md": envelope.revised_plan,
                "evidence_table.md": envelope.evidence_table,
            }
            json_outputs: dict[str, Any] = {
                "decision.json": serialized["decision"],
                "agent_status.json": self._agent_status(context, response),
            }
            if envelope.target_change is not None:
                json_outputs["target_change.json"] = serialized["target_change"]

            for name, content in text_outputs.items():
                temporary_paths[name].write_text(content, encoding="utf-8")
            temporary_paths["metric_candidates.yaml"].write_text(
                self._stable_yaml({"metrics": serialized["metric_candidates"]}),
                encoding="utf-8",
            )
            for name, content in json_outputs.items():
                temporary_paths[name].write_text(self._stable_json(content), encoding="utf-8")

            output_checksums = {
                name: sha256_file(temporary_paths[name])
                for name in output_names
                if name != "retriever_output_manifest.json"
            }
            manifest = self._manifest(
                context,
                material_manifest,
                response,
                output_names,
                output_checksums,
            )
            temporary_paths["retriever_output_manifest.json"].write_text(
                self._stable_json(manifest), encoding="utf-8"
            )
            self._revalidate_serialized_outputs(temporary_paths, context)

            for name, final_path in final_paths.items():
                if not final_path.exists():
                    continue
                backup_path = backup_root / name
                os.replace(final_path, backup_path)
                backup_paths[final_path] = backup_path

            for name in output_names:
                os.replace(temporary_paths[name], final_paths[name])
                published_paths.append(final_paths[name])
        except Exception as original_error:
            rollback_errors: list[tuple[str, Exception]] = []
            for path in reversed(published_paths):
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
                except Exception as rollback_error:
                    rollback_errors.append((f"delete published file {path}", rollback_error))
            for final_path, backup_path in reversed(tuple(backup_paths.items())):
                try:
                    os.replace(backup_path, final_path)
                except Exception as rollback_error:
                    rollback_errors.append((f"restore original file {final_path}", rollback_error))
            if rollback_errors:
                details = "; ".join(f"{operation}: {error}" for operation, error in rollback_errors)
                raise AgentProtocolError(
                    f"retriever publish failed; rollback incomplete: {details}"
                ) from original_error
            if temporary_root is not None:
                try:
                    shutil.rmtree(temporary_root)
                except Exception as cleanup_error:
                    raise AgentProtocolError(
                        "retriever publish failed; rollback succeeded but staging cleanup failed"
                    ) from cleanup_error
            raise

        if temporary_root is not None:
            try:
                shutil.rmtree(temporary_root)
            except Exception as cleanup_error:
                raise AgentProtocolError(
                    "retriever publish completed but staging cleanup failed"
                ) from cleanup_error
        return AgentResult.from_artifact_tree(
            context,
            stage=self.stage,
            status="completed",
            created_files=[final_paths[name] for name in output_names],
            warnings=(
                "metric candidates and target changes are advisory and require human approval",
            ),
            human_review_questions=self._human_review_questions(envelope),
        )

    @staticmethod
    def _stable_json(payload: Any) -> str:
        return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    @staticmethod
    def _stable_yaml(payload: Any) -> str:
        return yaml.safe_dump(payload, allow_unicode=True, sort_keys=True)

    def _agent_status(self, context: RunContext, response: Any) -> dict[str, Any]:
        return {
            "agent": self.name,
            "execution": "structured_provider_response",
            "required_outputs": list(self.required_outputs),
            "stage": self.stage,
            "status": "completed",
            "target": context.current_target.model_dump(mode="json", by_alias=True),
            "provider": {
                "model": response.model,
                "provider_request_id": response.provider_request_id,
            },
        }

    def _manifest(
        self,
        context: RunContext,
        material_manifest: MaterialManifest,
        response: Any,
        output_names: list[str],
        output_checksums: dict[str, str],
    ) -> dict[str, Any]:
        return {
            "agent": self.name,
            "complete": True,
            "run_id": context.run.run_id,
            "stage": self.stage,
            "status": "completed",
            "response_model": "RetrieverOutputEnvelope",
            "target": context.current_target.model_dump(mode="json", by_alias=True),
            "target_id": context.current_target.target_id,
            "target_revision": context.current_target.revision,
            "target_evidence_checksum": context.current_target.evidence_checksum,
            "material_manifest_checksum": material_manifest.manifest_checksum,
            "source_material_manifest_checksum": material_manifest.manifest_checksum,
            "provider": {
                "model": response.model,
                "provider_request_id": response.provider_request_id,
                "usage": response.usage,
            },
            "provider_model": response.model,
            "provider_request_id": response.provider_request_id,
            "outputs": output_names,
            "output_checksums": output_checksums,
        }

    def _revalidate_serialized_outputs(
        self, paths: dict[str, Path], context: RunContext
    ) -> None:
        try:
            payload = {
                "feasibility_review": paths["feasibility_review.md"].read_text(encoding="utf-8"),
                "revised_plan": paths["revised_plan.md"].read_text(encoding="utf-8"),
                "evidence_table": paths["evidence_table.md"].read_text(encoding="utf-8"),
                "metric_candidates": yaml.safe_load(
                    paths["metric_candidates.yaml"].read_text(encoding="utf-8")
                )["metrics"],
                "decision": json.loads(paths["decision.json"].read_text(encoding="utf-8")),
                "target_change": (
                    json.loads(paths["target_change.json"].read_text(encoding="utf-8"))
                    if "target_change.json" in paths
                    else None
                ),
            }
            envelope = RetrieverOutputEnvelope.model_validate(payload)
        except (
            OSError,
            UnicodeError,
            json.JSONDecodeError,
            TypeError,
            KeyError,
            yaml.YAMLError,
            ValidationError,
        ) as exc:
            raise AgentProtocolError(
                "serialized retriever artifacts do not match the retriever output schema"
            ) from exc
        if envelope.target_change is not None:
            self.validate_target_change(envelope.target_change, context.current_target)

    @staticmethod
    def _human_review_questions(envelope: RetrieverOutputEnvelope) -> tuple[str, ...]:
        questions = list(envelope.decision.human_questions)
        questions.append(
            "Human approval is required before using metric candidates for formal comparison."
        )
        if envelope.target_change is not None:
            questions.append(
                "Human approval is required before applying the proposed target change."
            )
        return tuple(questions)
