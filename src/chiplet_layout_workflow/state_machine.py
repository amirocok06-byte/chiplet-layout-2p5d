from __future__ import annotations

from pathlib import Path

from .artifacts.checksums import checksum_tree
from .artifacts.run_store import Run
from .checkpoints import (
    CHECKPOINT_STAGES,
    Checkpoint,
    CheckpointIntegrityError,
    read_all_checkpoints,
    read_checkpoint,
    verify_checkpoint_artifacts,
    write_checkpoint,
)
from .gates.human_gate import GateResult, HumanGate


STAGES = CHECKPOINT_STAGES
_DECISIONS = {"approve", "approved", "revise", "reject"}


class InvalidTransition(ValueError):
    """Raised when a workflow stage tries to skip the sequential protocol."""


class WorkflowStateMachine:
    """Five-stage, checkpoint-backed workflow state machine."""

    def __init__(
        self,
        run_or_root: Run | Path | str,
        *,
        gate: HumanGate | None = None,
    ) -> None:
        self.run = run_or_root if isinstance(run_or_root, Run) else None
        self.root = run_or_root.path if isinstance(run_or_root, Run) else Path(run_or_root)
        self.gate = gate or HumanGate()
        self._current_stage: str | None = None
        self._completed_stages: tuple[str, ...] = ()
        self._started = False
        self.last_gate_result: GateResult | None = None

    @property
    def current_stage(self) -> str | None:
        return self._current_stage

    @property
    def completed_stages(self) -> tuple[str, ...]:
        return self._completed_stages

    def start(self) -> "WorkflowStateMachine":
        if self._started:
            return self
        self.resume()
        self._started = True
        return self

    def resume(self) -> "WorkflowStateMachine":
        checkpoints = read_all_checkpoints(self.run or self.root)
        by_stage: dict[str, Checkpoint] = {}
        for checkpoint in checkpoints:
            if checkpoint.stage in by_stage:
                raise CheckpointIntegrityError(
                    f"multiple checkpoint records for stage '{checkpoint.stage}'"
                )
            verify_checkpoint_artifacts(checkpoint, self._artifact_directory(checkpoint.stage))
            by_stage[checkpoint.stage] = checkpoint

        completed: list[str] = []
        current: str | None = None
        for index, stage in enumerate(STAGES):
            checkpoint = by_stage.get(stage)
            if checkpoint is None:
                current = stage
                if any(later in by_stage for later in STAGES[index + 1 :]):
                    raise CheckpointIntegrityError(
                        f"checkpoint sequence skips required stage '{stage}'"
                    )
                break
            if checkpoint.decision == "approve":
                completed.append(stage)
                continue
            current = stage
            if any(later in by_stage for later in STAGES[index + 1 :]):
                raise CheckpointIntegrityError(
                    f"checkpoint sequence advances after non-approved stage '{stage}'"
                )
            break
        else:
            current = None

        self._completed_stages = tuple(completed)
        self._current_stage = current
        self._started = True
        return self

    def start_stage(self, stage: str) -> str:
        self._require_started()
        if stage not in STAGES:
            raise InvalidTransition(f"unknown workflow stage: {stage!r}")
        if stage != self._current_stage:
            raise InvalidTransition(
                f"stage '{stage}' cannot start while current stage is {self._current_stage!r}"
            )
        if stage != STAGES[0] and STAGES[STAGES.index(stage) - 1] not in self._completed_stages:
            raise InvalidTransition(f"stage '{stage}' requires an approved previous stage")
        return stage

    def request_gate(self, summary: str) -> GateResult:
        self._require_started()
        if self._current_stage is None:
            raise InvalidTransition("workflow is already complete")
        result = self.gate.request(self._current_stage, summary)
        self.last_gate_result = result
        return result

    def manifest_metadata(self) -> dict[str, bool]:
        """Return gate state fields that callers can merge into a manifest."""
        return self.gate.manifest_metadata()

    def complete_stage(
        self,
        stage: str,
        decision: str | GateResult,
        *,
        artifact_checksums: dict[str, str] | None = None,
        operator_note: str = "",
        timestamp: str | None = None,
    ) -> Checkpoint:
        self._require_started()
        self.start_stage(stage)
        if isinstance(decision, GateResult):
            if decision.decision is None:
                raise InvalidTransition("a disabled human gate cannot advance the workflow")
            operator_note = decision.operator_note or operator_note
            decision = decision.decision
        if decision not in _DECISIONS:
            raise InvalidTransition("decision must be approve, revise, or reject")
        canonical_decision = "approve" if decision == "approved" else decision
        gate_was_disabled = (
            not self.gate.enabled
            or (
                self.last_gate_result is not None
                and self.last_gate_result.status == "disabled"
                and not self.last_gate_result.enabled
            )
        )
        if canonical_decision == "approve" and gate_was_disabled:
            raise InvalidTransition("a disabled human gate cannot advance the workflow")
        if artifact_checksums is None:
            artifact_checksums = checksum_tree(self._artifact_directory(stage))
        path = write_checkpoint(
            self.run or self.root,
            stage,
            canonical_decision,
            artifact_checksums,
            timestamp=timestamp,
            operator_note=operator_note,
        )
        checkpoint = read_checkpoint(path)
        if canonical_decision == "approve":
            index = STAGES.index(stage)
            self._completed_stages = tuple(STAGES[: index + 1])
            self._current_stage = STAGES[index + 1] if index + 1 < len(STAGES) else None
        return checkpoint

    def advance(self) -> str | None:
        self._require_started()
        if self._current_stage is None:
            return None
        current_index = STAGES.index(self._current_stage)
        current_checkpoint_exists = any(
            checkpoint.stage == self._current_stage
            for checkpoint in read_all_checkpoints(self.run or self.root)
        )
        if (
            current_index > 0
            and STAGES[current_index - 1] in self._completed_stages
            and not current_checkpoint_exists
        ):
            return self._current_stage
        raise InvalidTransition(
            f"stage '{self._current_stage}' has not received an approved checkpoint"
        )

    def _require_started(self) -> None:
        if not self._started:
            raise InvalidTransition("workflow must be started before a stage can change")

    def _artifact_directory(self, stage: str) -> Path:
        if self.run is not None:
            names = {
                "reading": self.run.paper_understanding,
                "feasibility_review": self.run.feasibility,
                "building": self.run.source,
                "verification": self.run.verification,
                "version_archiving": self.run.versioning,
            }
            return names[stage]
        aliases = {
            "reading": ("reading", "paper_understanding"),
            "feasibility_review": ("feasibility_review", "feasibility"),
            "building": ("building", "source"),
            "verification": ("verification",),
            "version_archiving": ("version_archiving", "versioning"),
        }
        for name in aliases[stage]:
            candidate = self.root / name
            if candidate.exists():
                return candidate
        return self.root / aliases[stage][0]
