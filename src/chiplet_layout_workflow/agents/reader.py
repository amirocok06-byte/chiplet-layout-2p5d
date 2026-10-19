from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from ..paper_corpus import (
    Ai42CorpusManifest,
    CORPUS_MANIFEST_NAME,
    PaperCorpusError,
    read_ai42_5dic_corpus_manifest,
    reader_artifacts_from_corpus,
)
from ..reader_outputs import ReaderOutputEnvelope, reader_provider_response_schema
from ..target_policy import AI42_5DIC_TARGET_ID
from .base import AgentProtocolError, AgentResult, BaseAgent, RunContext


class ReaderAgent(BaseAgent):
    name = "reader"
    role = "reader"
    stage = "reading"
    required_outputs = (
        "paper_understanding.md",
        "core_method.md",
        "transferable_ideas.md",
        "limitations.md",
        "citations.json",
        "candidate_plan.md",
        "target_assessment.json",
    )

    _SUBSTANTIVE_OUTPUTS = (
        "paper_understanding.md",
        "core_method.md",
        "candidate_plan.md",
        "transferable_ideas.md",
        "limitations.md",
        "citations.json",
        "target_assessment.json",
    )
    _METADATA_OUTPUTS = ("agent_status.json", "reader_output_manifest.json")

    @staticmethod
    def _corpus_manifest(
        context: RunContext,
    ) -> tuple[Path, Ai42CorpusManifest] | None:
        all_staged_paths = context._all_staged_input_paths()
        corpus_manifests = [
            path for path in all_staged_paths if path.name == CORPUS_MANIFEST_NAME
        ]
        if not corpus_manifests:
            return None
        if len(corpus_manifests) != 1:
            raise AgentProtocolError("reader requires exactly one corpus manifest")
        manifest_path = corpus_manifests[0]
        try:
            manifest = read_ai42_5dic_corpus_manifest(
                manifest_path, context.run.inputs
            )
        except PaperCorpusError as exc:
            raise AgentProtocolError("reader corpus validation failed") from exc
        if manifest.corpus_checksum != context.current_target.evidence_checksum:
            raise AgentProtocolError(
                "reader corpus checksum does not match the approved target"
            )
        return manifest_path, manifest

    @classmethod
    def _validated_evidence(cls, context: RunContext) -> list[Path]:
        corpus_info = cls._corpus_manifest(context)
        if corpus_info is not None:
            manifest_path, _manifest = corpus_info
            return list(reader_artifacts_from_corpus(manifest_path, context.run.inputs))

        if context.current_target.target_id == AI42_5DIC_TARGET_ID and context.input_paths:
            raise AgentProtocolError(
                "ai42.5Dic Reader requires a verified corpus manifest; "
                "explicit input_paths are not accepted in legacy mode"
            )

        evidence_paths = context.evidence_paths()

        allowed_suffixes = {".md", ".markdown", ".txt", ".rst"}
        valid_evidence: list[Path] = []
        for path in evidence_paths:
            try:
                if path.suffix.lower() in allowed_suffixes and path.stat().st_size > 0:
                    text = path.read_bytes().decode("utf-8")
                    if "\x00" in text:
                        continue
                    valid_evidence.append(path)
            except (OSError, UnicodeDecodeError):
                continue
        return valid_evidence

    def validate_inputs(self, context: RunContext) -> list[Path]:
        valid_evidence = self._validated_evidence(context)
        if not valid_evidence:
            raise ValueError("reader requires at least one paper or note input")
        return valid_evidence

    def run(self, context: RunContext) -> AgentResult:
        context = self._stage_context(context)
        corpus_info = self._corpus_manifest(context)
        evidence = self.validate_inputs(context)

        prompt = context.render_role_prompt(self.role, stage=self.stage)
        response = self.provider.generate(
            prompt,
            prompt,
            evidence,
            response_schema=reader_provider_response_schema(),
        )
        envelope = self._parse_response(response)
        self._validate_target_assessment(envelope, context)
        self._validate_citations(
            envelope, corpus_info[1] if corpus_info is not None else None
        )
        return self._publish(context, envelope)

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
    def _parse_response(response: Any) -> ReaderOutputEnvelope:
        try:
            payload = json.loads(response.text)
        except (AttributeError, TypeError, json.JSONDecodeError) as exc:
            raise AgentProtocolError("reader provider returned invalid JSON") from exc

        try:
            return ReaderOutputEnvelope.model_validate(payload)
        except (TypeError, ValidationError) as exc:
            raise AgentProtocolError(
                "reader provider response does not match the reader output schema"
            ) from exc

    @staticmethod
    def _validate_target_assessment(
        envelope: ReaderOutputEnvelope, context: RunContext
    ) -> None:
        assessment = envelope.target_assessment
        current_target_id = context.current_target.target_id
        if assessment.current_target_id != current_target_id:
            raise AgentProtocolError(
                "reader target assessment does not match the current target"
            )
        if assessment.proposal is not None and assessment.proposal.target_id == current_target_id:
            raise AgentProtocolError(
                "reader target proposal must differ from the current target"
            )

    @staticmethod
    def _validate_citations(
        envelope: ReaderOutputEnvelope,
        corpus_manifest: Ai42CorpusManifest | None,
    ) -> None:
        if corpus_manifest is None:
            return
        valid_sources: dict[str, set[tuple[str, str]]] = {}
        for paper in corpus_manifest.papers:
            valid_sources.setdefault(paper.source_relative_path, set()).add(
                ("reference_paper", paper.source_sha256)
            )
        for note in corpus_manifest.ai_placement_notes:
            valid_sources.setdefault(note.source_relative_path, set()).add(
                ("ai_placement_note", note.source_sha256)
            )
        for citation in envelope.citations:
            expected = valid_sources.get(citation.source)
            if expected is None or (
                citation.source_kind,
                citation.source_sha256,
            ) not in expected:
                raise AgentProtocolError(
                    "reader citation is not bound to the verified corpus manifest"
                )
    def _publish(
        self, context: RunContext, envelope: ReaderOutputEnvelope
    ) -> AgentResult:
        root = context.artifact_root(self.stage)
        output_names = self._SUBSTANTIVE_OUTPUTS + self._METADATA_OUTPUTS
        final_paths = {
            name: context.artifact_path(self.stage, name) for name in output_names
        }
        published_paths: list[Path] = []
        backup_paths: dict[Path, Path] = {}
        temporary_root: Path | None = None

        try:
            temporary_root = Path(
                tempfile.mkdtemp(prefix=".reader-", dir=str(root))
            )
            context.run.assert_inside(temporary_root)
            temporary_relative = temporary_root.relative_to(root)
            temporary_paths = {
                name: context.artifact_path(
                    self.stage, temporary_relative / name
                )
                for name in output_names
            }
            backup_root = context.artifact_path(
                self.stage, temporary_relative / "backup"
            )
            backup_root.mkdir()
            for name, final_path in final_paths.items():
                if not final_path.exists():
                    continue
                backup_path = context.artifact_path(
                    self.stage, temporary_relative / "backup" / name
                )
                os.replace(final_path, backup_path)
                backup_paths[final_path] = backup_path

            serialized = envelope.model_dump(mode="json", by_alias=True)
            text_outputs = {
                "paper_understanding.md": envelope.paper_understanding,
                "core_method.md": envelope.core_method,
                "candidate_plan.md": envelope.candidate_plan,
                "transferable_ideas.md": envelope.transferable_ideas,
                "limitations.md": envelope.limitations,
            }
            json_outputs = {
                "citations.json": serialized["citations"],
                "target_assessment.json": serialized["target_assessment"],
                "agent_status.json": self._agent_status(context),
                "reader_output_manifest.json": self._manifest(context),
            }
            for name, content in text_outputs.items():
                temporary_paths[name].write_text(content, encoding="utf-8")
            for name, content in json_outputs.items():
                temporary_paths[name].write_text(
                    self._stable_json(content), encoding="utf-8"
                )

            self._revalidate_serialized_outputs(temporary_paths, context)

            for name in output_names[:-1]:
                os.replace(temporary_paths[name], final_paths[name])
                published_paths.append(final_paths[name])
            os.replace(
                temporary_paths["reader_output_manifest.json"],
                final_paths["reader_output_manifest.json"],
            )
            published_paths.append(final_paths["reader_output_manifest.json"])

        except Exception as original_error:
            rollback_errors: list[tuple[str, Exception]] = []
            for path in reversed(published_paths):
                try:
                    path.unlink()
                except Exception as rollback_error:
                    rollback_errors.append(
                        (f"delete published file {path}", rollback_error)
                    )
            for final_path, backup_path in reversed(tuple(backup_paths.items())):
                try:
                    os.replace(backup_path, final_path)
                except Exception as rollback_error:
                    rollback_errors.append(
                        (f"restore original file {final_path}", rollback_error)
                    )
            if rollback_errors:
                details = "; ".join(
                    f"{operation}: {error}" for operation, error in rollback_errors
                )
                raise AgentProtocolError(
                    f"reader publish failed; rollback incomplete: {details}"
                ) from original_error
            if temporary_root is not None:
                try:
                    shutil.rmtree(temporary_root)
                except Exception as cleanup_error:
                    raise AgentProtocolError(
                        "reader publish failed; rollback succeeded but staging cleanup failed"
                    ) from cleanup_error
            raise

        if temporary_root is not None:
            try:
                shutil.rmtree(temporary_root)
            except Exception as cleanup_error:
                raise AgentProtocolError(
                    "reader publish completed but staging cleanup failed"
                ) from cleanup_error
        return AgentResult.from_artifact_tree(
            context,
            stage=self.stage,
            status="completed",
            created_files=[final_paths[name] for name in output_names],
            warnings=(
                "reader target assessment is advisory and does not change the approved target",
            ),
            human_review_questions=self._human_review_questions(envelope),
        )

    @staticmethod
    def _stable_json(payload: Any) -> str:
        return json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def _agent_status(self, context: RunContext) -> dict[str, Any]:
        return {
            "agent": self.name,
            "execution": "structured_provider_response",
            "required_outputs": list(self.required_outputs),
            "stage": self.stage,
            "status": "completed",
            "target": context.current_target.model_dump(mode="json", by_alias=True),
        }

    def _manifest(self, context: RunContext) -> dict[str, Any]:
        return {
            "agent": self.name,
            "complete": True,
            "metadata_outputs": list(self._METADATA_OUTPUTS),
            "response_model": "ReaderOutputEnvelope",
            "stage": self.stage,
            "status": "completed",
            "substantive_outputs": list(self._SUBSTANTIVE_OUTPUTS),
            "target_id": context.current_target.target_id,
        }

    def _revalidate_serialized_outputs(
        self, paths: dict[str, Path], context: RunContext
    ) -> None:
        payload = {
            "paper_understanding": paths["paper_understanding.md"].read_text(
                encoding="utf-8"
            ),
            "core_method": paths["core_method.md"].read_text(encoding="utf-8"),
            "candidate_plan": paths["candidate_plan.md"].read_text(encoding="utf-8"),
            "transferable_ideas": paths["transferable_ideas.md"].read_text(
                encoding="utf-8"
            ),
            "limitations": paths["limitations.md"].read_text(encoding="utf-8"),
            "citations": json.loads(
                paths["citations.json"].read_text(encoding="utf-8")
            ),
            "target_assessment": json.loads(
                paths["target_assessment.json"].read_text(encoding="utf-8")
            ),
        }
        try:
            envelope = ReaderOutputEnvelope.model_validate(payload)
        except (TypeError, ValidationError) as exc:
            raise AgentProtocolError(
                "serialized reader artifacts do not match the reader output schema"
            ) from exc
        self._validate_target_assessment(envelope, context)

    @staticmethod
    def _human_review_questions(
        envelope: ReaderOutputEnvelope,
    ) -> tuple[str, ...]:
        assessment = envelope.target_assessment
        questions = [
            f"Manually confirm the reader target assessment ({assessment.status}) before downstream use."
        ]
        questions.extend(assessment.human_questions)
        if assessment.status in {"uncertain", "needs_change"}:
            questions.append(
                "Human approval is required before acting on any uncertain or proposed target change."
            )
        return tuple(questions)
