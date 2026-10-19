from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..artifacts.checksums import checksum_tree
from ..artifacts.run_store import (
    PathBoundaryError,
    Run,
    _reject_link_components,
)
from ..prompts import PromptRenderer
from ..providers import LLMProvider
from ..research_target import ResearchTarget
from ..target_policy import (
    AI42_5DIC_TARGET_ID,
    TargetPolicyError,
    require_matching_evidence,
    require_target_id,
)


STAGES = (
    "reading",
    "feasibility_review",
    "building",
    "verification",
    "version_archiving",
)


class AgentProtocolError(ValueError):
    """Raised when an agent request violates the stage protocol."""


class ArtifactBoundaryError(AgentProtocolError):
    """Raised when an agent attempts to address an unsafe artifact path."""


class GitAdapter(Protocol):
    """Task 6 boundary for a future, injected Git implementation."""

    def archive(
        self,
        artifact_paths: Sequence[Path],
        metadata: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        ...


@dataclass(frozen=True)
class RunContext:
    """Immutable stage context with a late-bound, approved research target."""

    run: Run
    current_target: ResearchTarget
    provider: LLMProvider
    prompt_renderer: PromptRenderer = field(default_factory=PromptRenderer)
    input_paths: tuple[Path, ...] = ()
    source_worktree: Path | None = None
    git_adapter: GitAdapter | None = None
    benchmark_executor: object | None = None
    stage: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        policy_metadata = dict(self.metadata)
        policy_metadata.setdefault("target_lock_id", AI42_5DIC_TARGET_ID)
        policy_metadata.setdefault("require_target_evidence", True)
        object.__setattr__(self, "metadata", policy_metadata)

        try:
            self.current_target.require_approved()
        except ValueError as exc:
            raise AgentProtocolError(
                "RunContext requires an approved current research target"
            ) from exc

        source_worktree = (
            self.source_worktree if self.source_worktree is not None else self.run.source
        )
        try:
            source_worktree = self.run.assert_inside(source_worktree)
            _reject_link_components(source_worktree)
            source_worktree.resolve(strict=False).relative_to(
                self.run.source.resolve(strict=True)
            )
        except (OSError, RuntimeError, ValueError, PathBoundaryError) as exc:
            raise ArtifactBoundaryError(
                "source_worktree must remain inside run.source and contain no links"
            ) from exc
        object.__setattr__(self, "source_worktree", source_worktree)

        normalized_inputs = tuple(
            self._validate_explicit_input_path(Path(path))
            for path in self.input_paths
        )
        object.__setattr__(self, "input_paths", normalized_inputs)
        if self.stage is not None and self.stage not in STAGES:
            raise AgentProtocolError(f"unknown context stage: {self.stage!r}")

        locked_target_id = self.metadata["target_lock_id"]
        if locked_target_id is not None:
            try:
                require_target_id(self.current_target, str(locked_target_id))
            except TargetPolicyError as exc:
                raise AgentProtocolError(str(exc)) from exc

        if self.metadata["require_target_evidence"] is True:
            try:
                require_matching_evidence(
                    self.current_target,
                    self._all_staged_input_paths(),
                    allowed_root=self.run.inputs,
                )
            except TargetPolicyError as exc:
                raise AgentProtocolError(str(exc)) from exc

    def _validate_explicit_input_path(self, path: Path) -> Path:
        candidate = path if path.is_absolute() else self.run.path / path
        try:
            self.run.assert_inside(candidate)
            _reject_link_components(candidate)
            candidate.resolve(strict=False).relative_to(
                self.run.inputs.resolve(strict=True)
            )
        except (OSError, RuntimeError, ValueError, PathBoundaryError) as exc:
            raise ArtifactBoundaryError(
                f"input path must remain in run.inputs and contain no links: {path!r}"
            ) from exc
        try:
            is_file = candidate.is_file()
        except OSError as exc:
            raise ArtifactBoundaryError(
                f"input path cannot be inspected safely: {path!r}"
            ) from exc
        if not is_file:
            raise ArtifactBoundaryError(
                f"input path must be an existing regular file: {path!r}"
            )
        return candidate

    @classmethod
    def for_run(
        cls,
        run: Run,
        *,
        provider: LLMProvider,
        current_target: ResearchTarget | None = None,
        **kwargs: Any,
    ) -> "RunContext":
        return cls(
            run=run,
            current_target=current_target or run.research_target,
            provider=provider,
            **kwargs,
        )

    def for_stage(self, stage: str) -> "RunContext":
        if stage not in STAGES:
            raise AgentProtocolError(f"unknown workflow stage: {stage!r}")
        if self.stage is not None and self.stage != stage:
            raise AgentProtocolError(
                f"context is bound to stage {self.stage!r}, not {stage!r}"
            )
        return RunContext(
            run=self.run,
            current_target=self.current_target,
            provider=self.provider,
            prompt_renderer=self.prompt_renderer,
            input_paths=self.input_paths,
            source_worktree=self.source_worktree,
            git_adapter=self.git_adapter,
            benchmark_executor=self.benchmark_executor,
            stage=stage,
            metadata=self.metadata,
        )

    def artifact_root(self, stage: str) -> Path:
        roots = {
            "reading": self.run.paper_understanding,
            "feasibility_review": self.run.feasibility,
            "building": self.run.source,
            "verification": self.run.verification,
            "version_archiving": self.run.versioning,
        }
        if stage not in roots:
            raise AgentProtocolError(f"unknown workflow stage: {stage!r}")
        if self.stage is not None and self.stage != stage:
            raise ArtifactBoundaryError(
                f"context is bound to stage {self.stage!r}, not {stage!r}"
            )
        root = roots[stage]
        try:
            self.run.assert_inside(root)
        except PathBoundaryError as exc:
            raise ArtifactBoundaryError(
                f"artifact root for stage {stage!r} is outside the run"
            ) from exc
        return root

    def artifact_path(self, stage: str, relative_path: str | Path) -> Path:
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ArtifactBoundaryError("artifact paths must be relative and stay in stage")
        root = self.artifact_root(stage)
        candidate = root / relative
        try:
            candidate = self.run.assert_inside(candidate)
            resolved = candidate.resolve(strict=False)
            resolved.relative_to(root.resolve(strict=True))
        except (OSError, RuntimeError, ValueError, PathBoundaryError) as exc:
            raise ArtifactBoundaryError(
                f"artifact path escapes stage {stage!r}: {relative_path!r}"
            ) from exc
        return candidate

    def run_relative(self, path: Path | str) -> str:
        try:
            candidate = self.run.assert_inside(path).resolve(strict=False)
            relative = candidate.relative_to(self.run.path.resolve(strict=True))
        except (OSError, RuntimeError, ValueError, PathBoundaryError) as exc:
            raise ArtifactBoundaryError("path must remain inside the current run") from exc
        return relative.as_posix()

    def render_role_prompt(self, role: str, *, stage: str) -> str:
        rendered = self.prompt_renderer.render(role, {"stage": stage})
        target = json.dumps(
            self.current_target.model_dump(mode="json", by_alias=True),
            ensure_ascii=False,
            sort_keys=True,
        )
        return (
            f"{rendered}\n\nCurrent research target (late-bound JSON): {target}\n"
            "Treat this target as injected context; do not infer or replace it from the topic."
        )

    def evidence_paths(self) -> tuple[Path, ...]:
        """Return read-only evidence paths without widening the output boundary."""
        if self.input_paths:
            return self.input_paths
        return self._all_staged_input_paths()

    def _all_staged_input_paths(self) -> tuple[Path, ...]:
        """Return every safe regular file under ``run.inputs``."""
        if not self.run.inputs.exists():
            return ()
        try:
            _reject_link_components(self.run.inputs)
        except (OSError, PathBoundaryError):
            return ()
        evidence: list[Path] = []
        for path in self.run.inputs.rglob("*"):
            if not path.is_file():
                continue
            try:
                evidence.append(self._validate_explicit_input_path(path))
            except ArtifactBoundaryError:
                continue
        return tuple(sorted(evidence))

    @property
    def target(self) -> ResearchTarget:
        """Short compatibility name for the injected current target."""
        return self.current_target


@dataclass(frozen=True)
class AgentResult:
    stage: str
    status: str
    created_files: tuple[str, ...]
    checksums: dict[str, str]
    warnings: tuple[str, ...] = ()
    human_review_questions: tuple[str, ...] = ()

    @classmethod
    def from_artifact_tree(
        cls,
        context: RunContext,
        *,
        stage: str,
        status: str,
        created_files: Sequence[Path | str],
        warnings: Sequence[str] = (),
        human_review_questions: Sequence[str] = (),
    ) -> "AgentResult":
        root = context.artifact_root(stage)
        result = cls(
            stage=stage,
            status=status,
            created_files=tuple(context.run_relative(path) for path in created_files),
            checksums=checksum_tree(root),
            warnings=tuple(warnings),
            human_review_questions=tuple(human_review_questions),
        )
        result.validate(context)
        return result

    def validate(self, context: RunContext) -> "AgentResult":
        if self.stage not in STAGES:
            raise AgentProtocolError(f"unknown result stage: {self.stage!r}")
        if context.stage is not None and self.stage != context.stage:
            raise ArtifactBoundaryError(
                f"result stage {self.stage!r} does not match context stage {context.stage!r}"
            )
        root = context.artifact_root(self.stage)
        actual_checksums = checksum_tree(root)
        for relative in self.created_files:
            path = context.run.path / relative
            try:
                path.resolve(strict=False).relative_to(root.resolve(strict=True))
            except (OSError, RuntimeError, ValueError) as exc:
                raise ArtifactBoundaryError(
                    f"created file is outside stage {self.stage!r}: {relative!r}"
                ) from exc
        if dict(self.checksums) != actual_checksums:
            raise AgentProtocolError(
                f"result checksums do not match stage {self.stage!r} artifact tree"
            )
        return self

    def as_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "status": self.status,
            "created_files": list(self.created_files),
            "checksums": dict(self.checksums),
            "warnings": list(self.warnings),
            "human_review_questions": list(self.human_review_questions),
        }


class Agent(Protocol):
    name: str
    role: str
    stage: str
    required_outputs: tuple[str, ...]

    def run(self, context: RunContext) -> AgentResult:
        ...


class BaseAgent:
    """Shared protocol shell; domain execution is intentionally deferred."""

    name = "base"
    role = "base"
    stage = ""
    required_outputs: tuple[str, ...] = ()

    def __init__(self, provider: LLMProvider, *, prompt_renderer: PromptRenderer | None = None):
        self.provider = provider
        self.prompt_renderer = prompt_renderer or PromptRenderer()

    def run(self, context: RunContext) -> AgentResult:
        if context.stage is not None and context.stage != self.stage:
            context.for_stage(self.stage)
        if context.provider is not self.provider:
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
        validation_context = context
        if context.stage is not None:
            validation_context = RunContext(
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
        self.validate_inputs(validation_context)
        context = context.for_stage(self.stage)
        prompt = context.render_role_prompt(self.role, stage=self.stage)
        self.provider.generate(prompt, prompt, list(context.evidence_paths()))

        status_path = context.artifact_path(self.stage, "agent_status.json")
        status_path.parent.mkdir(parents=True, exist_ok=True)
        status_path.write_text(
            json.dumps(
                {
                    "agent": self.name,
                    "stage": self.stage,
                    "status": "scaffolded",
                    "target": context.current_target.model_dump(mode="json", by_alias=True),
                    "required_outputs": list(self.required_outputs),
                    "execution": "full stage execution is intentionally deferred",
                },
                indent=2,
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return AgentResult.from_artifact_tree(
            context,
            stage=self.stage,
            status="scaffolded",
            created_files=[status_path],
            warnings=["full domain execution is not implemented in this scaffold"],
            human_review_questions=[
                f"Approve the {self.stage} stage and injected research target?"
            ],
        )

    def validate_inputs(self, context: RunContext) -> None:
        del context
