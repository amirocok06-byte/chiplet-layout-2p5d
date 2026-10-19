from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .agents import (
    Agent,
    AgentResult,
    AgentProtocolError,
    BuilderAgent,
    ReaderAgent,
    RetrieverAgent,
    RunContext,
    VerifierAgent,
    VersionControllerAgent,
)
from .artifacts.checksums import checksum_tree
from .artifacts.run_store import Run
from .gates import GateResult, HumanGate
from .handoff_receipts import write_builder_approval_receipt
from .providers import LLMProvider, MockProvider
from .research_target import ResearchTarget
from .state_machine import WorkflowStateMachine
from .target_policy import AI42_5DIC_TARGET_ID, TargetPolicyError, require_target_id


@dataclass(frozen=True)
class OrchestrationResult:
    status: str
    stage: str | None
    agent_result: AgentResult | None
    gate_result: GateResult | None
    message: str = ""


class WorkflowOrchestrator:
    """Sequential, one-stage-at-a-time workflow skeleton.

    This class wires protocol and gate boundaries only. Domain analysis, source
    construction, benchmark execution, metric comparison, and real Git writes
    are intentionally left for later tasks. The current production entry point
    is locked to ``ai42.5Dic`` and requires matching staged evidence unless a
    caller explicitly supplies a different target policy for isolated tests or
    a future, human-approved topic.
    """

    def __init__(
        self,
        run: Run,
        *,
        current_target: ResearchTarget | None = None,
        provider: LLMProvider | None = None,
        reader: Agent | None = None,
        retriever: Agent | None = None,
        builder: Agent | None = None,
        verifier: Agent | None = None,
        version_controller: Agent | None = None,
        gate: HumanGate | None = None,
        context_options: dict[str, Any] | None = None,
        target_lock_id: str | None = AI42_5DIC_TARGET_ID,
        require_target_evidence: bool = True,
    ) -> None:
        selected_provider = provider or self._agent_provider(reader) or MockProvider({})
        selected_target = current_target or run.research_target
        if target_lock_id is not None:
            try:
                require_target_id(selected_target, target_lock_id)
            except TargetPolicyError as exc:
                raise AgentProtocolError(str(exc)) from exc
        options = dict(context_options or {})
        metadata = dict(options.get("metadata", {}))
        metadata["target_lock_id"] = target_lock_id
        metadata["require_target_evidence"] = require_target_evidence
        options["metadata"] = metadata
        self.context = RunContext(
            run=run,
            current_target=selected_target,
            provider=selected_provider,
            **options,
        )
        self.agents: dict[str, Agent] = {
            "reading": reader or ReaderAgent(selected_provider),
            "feasibility_review": retriever or RetrieverAgent(selected_provider),
            "building": builder or BuilderAgent(selected_provider),
            "verification": verifier or VerifierAgent(selected_provider),
            "version_archiving": version_controller
            or VersionControllerAgent(selected_provider),
        }
        self.gate = gate or HumanGate()
        self.state_machine = WorkflowStateMachine(run, gate=self.gate)

    @staticmethod
    def _agent_provider(agent: Agent | None) -> LLMProvider | None:
        return getattr(agent, "provider", None) if agent is not None else None

    def run(self) -> OrchestrationResult:
        """Execute at most the current stage and return its gate outcome."""
        self.state_machine.start()
        stage = self.state_machine.current_stage
        if stage is None:
            return OrchestrationResult(
                status="complete",
                stage=None,
                agent_result=None,
                gate_result=None,
                message="all currently wired stages are already approved",
            )

        agent = self.agents[stage]
        if getattr(agent, "stage", None) != stage:
            raise AgentProtocolError(
                f"injected agent stage {getattr(agent, 'stage', None)!r} "
                f"does not match current stage {stage!r}"
            )
        context = self.context.for_stage(stage)
        agent_result = agent.run(context)
        if not isinstance(agent_result, AgentResult):
            raise AgentProtocolError("agent returned an invalid AgentResult")
        agent_result.validate(context)
        summary = self._gate_summary(agent_result)
        gate_result = self.state_machine.request_gate(summary)
        if gate_result.decision != "approve":
            return OrchestrationResult(
                status="paused",
                stage=stage,
                agent_result=agent_result,
                gate_result=gate_result,
                message=(
                    "stage remains paused until an enabled human gate approves it"
                ),
            )

        if stage == "feasibility_review":
            write_builder_approval_receipt(self.context, gate_result)

        self.state_machine.complete_stage(
            stage,
            gate_result,
            artifact_checksums=(
                checksum_tree(context.artifact_root(stage))
                if stage == "feasibility_review"
                else agent_result.checksums
            ),
            operator_note=gate_result.operator_note,
        )
        return OrchestrationResult(
            status="approved",
            stage=stage,
            agent_result=agent_result,
            gate_result=gate_result,
            message="stage checkpoint approved; call run() again for the next stage",
        )

    @staticmethod
    def _gate_summary(result: AgentResult) -> str:
        files = ", ".join(result.created_files) or "no files"
        questions = "; ".join(result.human_review_questions) or "none"
        return (
            f"status={result.status}; created_files={files}; "
            f"human_review_questions={questions}"
        )
