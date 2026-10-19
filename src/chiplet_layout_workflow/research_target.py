from __future__ import annotations

from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator


class _TargetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_non_empty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _reject_duplicate_target_aliases(value: object) -> object:
    if isinstance(value, dict) and "id" in value and "target_id" in value:
        raise ValueError("provide only one of id or target_id")
    return value


class ResearchTarget(_TargetModel):
    """The currently selected research target, including its audit metadata."""

    target_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("target_id", "id"),
        serialization_alias="id",
    )
    display_name: str = Field(min_length=1)
    revision: int = Field(ge=0)
    source: str = Field(min_length=1)
    evidence_checksum: str = Field(min_length=1)
    approved: bool
    approval_receipt: "TargetApprovalReceipt | None" = None

    @model_validator(mode="before")
    @classmethod
    def reject_duplicate_aliases(cls, value: object) -> object:
        return _reject_duplicate_target_aliases(value)

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, value: str) -> str:
        return _require_non_empty(value, "target_id")

    @field_validator("display_name", "source", "evidence_checksum")
    @classmethod
    def validate_metadata_strings(cls, value: str, info) -> str:
        return _require_non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_approval_provenance(self) -> "ResearchTarget":
        if self.approved and self.approval_receipt is None:
            raise ValueError("approved research targets require an approval receipt")
        if not self.approved and self.approval_receipt is not None:
            raise ValueError("unapproved research targets cannot carry an approval receipt")
        if self.approval_receipt is not None and not self.approval_receipt.matches_target(self):
            raise ValueError("approval receipt does not match research target")
        return self

    @classmethod
    def seed(cls, target_id: str, display_name: str) -> "ResearchTarget":
        return cls(
            target_id=target_id,
            display_name=display_name,
            revision=0,
            source="user_seed",
            evidence_checksum="seed",
            approved=False,
        )

    @classmethod
    def from_approval(
        cls,
        proposal: "TargetProposal",
        decision: "TargetDecision",
        *,
        revision: int = 0,
    ) -> "ResearchTarget":
        receipt = TargetApprovalReceipt.issue(proposal, decision)
        return cls(
            target_id=proposal.target_id,
            display_name=proposal.display_name,
            revision=revision,
            source=proposal.source,
            evidence_checksum=proposal.evidence_checksum,
            approved=True,
            approval_receipt=receipt,
        )

    @classmethod
    def default_seed(cls) -> "ResearchTarget":
        # The active workflow topic is fixed by the user to ai42.5Dic. The
        # seed remains unapproved until a human binds the real paper evidence.
        return cls.seed("ai42.5Dic", "ai42.5Dic")

    @property
    def id(self) -> str:
        """Compatibility accessor for serialized ``id`` terminology."""
        return self.target_id

    @property
    def is_seed(self) -> bool:
        return self.source == "user_seed" and self.revision == 0 and not self.approved

    def require_approved(self) -> "ResearchTarget":
        receipt = getattr(self, "approval_receipt", None)
        if not self.approved or not isinstance(receipt, TargetApprovalReceipt):
            raise ValueError("an approved research target is required")
        if not receipt.matches_target(self):
            raise ValueError("research target approval provenance is invalid")
        return self


class TargetProposal(_TargetModel):
    """A candidate target that cannot become active without a human decision."""

    target_id: str = Field(
        min_length=1,
        validation_alias=AliasChoices("target_id", "id"),
        serialization_alias="id",
    )
    display_name: str = Field(min_length=1)
    source: str = Field(default="paper_evidence", min_length=1)
    evidence_checksum: str = Field(min_length=1)
    approved: bool = False
    requires_human_approval: Literal[True] = True
    approval_state: Literal["pending", "approved", "rejected"] = "pending"

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, value: str) -> str:
        return _require_non_empty(value, "target_id")

    @field_validator("display_name", "source", "evidence_checksum")
    @classmethod
    def validate_metadata_strings(cls, value: str, info) -> str:
        return _require_non_empty(value, info.field_name)

    @model_validator(mode="after")
    def validate_approval_state(self) -> "TargetProposal":
        if self.approved and self.approval_state != "approved":
            raise ValueError("approved proposals must have approval_state='approved'")
        if self.approval_state == "approved" and not self.approved:
            raise ValueError("approval_state='approved' requires approved=True")
        return self

    @classmethod
    def from_paper_evidence(
        cls,
        *,
        target_id: str,
        display_name: str,
        evidence_checksum: str,
        source: str = "paper_evidence",
    ) -> "TargetProposal":
        return cls(
            target_id=target_id,
            display_name=display_name,
            source=source,
            evidence_checksum=evidence_checksum,
            approved=False,
            requires_human_approval=True,
            approval_state="pending",
        )


class TargetDecision(_TargetModel):
    """The human gate decision associated with a target proposal."""

    target_id: str | None = Field(
        default=None,
        min_length=1,
        validation_alias=AliasChoices("target_id", "id"),
        serialization_alias="id",
    )
    approved: bool = False
    evidence_checksum: str | None = Field(default=None, min_length=1)
    source: str | None = Field(default=None, min_length=1)
    decision: Literal["approve", "reject"] | None = None
    proposal: TargetProposal | None = None

    @field_validator("target_id")
    @classmethod
    def validate_target_id(cls, value: str | None) -> str | None:
        return None if value is None else _require_non_empty(value, "target_id")

    @field_validator("evidence_checksum", "source")
    @classmethod
    def validate_optional_metadata(cls, value: str | None, info) -> str | None:
        return None if value is None else _require_non_empty(value, info.field_name)

    @model_validator(mode="after")
    def fill_from_proposal_and_validate(self) -> "TargetDecision":
        if self.proposal is not None:
            if self.target_id is not None and self.target_id != self.proposal.target_id:
                raise ValueError("target decision target_id does not match proposal")
            if (
                self.evidence_checksum is not None
                and self.evidence_checksum != self.proposal.evidence_checksum
            ):
                raise ValueError("target decision evidence_checksum does not match proposal")
            if self.source is not None and self.source != self.proposal.source:
                raise ValueError("target decision source does not match proposal")
            if self.proposal.approved and not self.approved:
                raise ValueError("target decision approval does not match proposal")
            if self.proposal.approval_state == "rejected" and self.approved:
                raise ValueError("target decision approval does not match proposal")
            if self.target_id is None:
                self.target_id = self.proposal.target_id
            if self.evidence_checksum is None:
                self.evidence_checksum = self.proposal.evidence_checksum
            if self.source is None:
                self.source = self.proposal.source
        if self.target_id is None:
            raise ValueError("target decision requires target_id")
        if self.approved and not self.evidence_checksum:
            raise ValueError("approved target decisions require evidence_checksum")
        expected_decision = "approve" if self.approved else "reject"
        if self.decision is None:
            self.decision = expected_decision
        elif self.decision != expected_decision:
            raise ValueError("decision does not match approved state")
        return self

    @classmethod
    def approve(cls, proposal: TargetProposal) -> "TargetDecision":
        return cls(proposal=proposal, approved=True, decision="approve")

    @classmethod
    def reject(cls, proposal: TargetProposal) -> "TargetDecision":
        return cls(proposal=proposal, approved=False, decision="reject")


class TargetApprovalReceipt(_TargetModel):
    """Persistable provenance proving which proposal a human approved."""

    receipt_version: Literal[1] = 1
    proposal: TargetProposal
    decision: TargetDecision

    @model_validator(mode="after")
    def validate_binding(self) -> "TargetApprovalReceipt":
        if not self.decision.approved or self.decision.decision != "approve":
            raise ValueError("approval receipt requires an approved target decision")
        if self.decision.proposal is None:
            raise ValueError("approval receipt requires a decision bound to a proposal")
        if self.decision.proposal.model_dump(mode="python") != self.proposal.model_dump(mode="python"):
            raise ValueError("approval receipt decision is bound to a different proposal")
        return self

    @classmethod
    def issue(cls, proposal: TargetProposal, decision: TargetDecision) -> "TargetApprovalReceipt":
        if decision.proposal is None:
            raise ValueError("approval receipt requires a decision bound to a proposal")
        if decision.proposal is not proposal:
            raise ValueError("approval receipt decision is bound to a different proposal")
        return cls(proposal=proposal, decision=decision)

    def matches_target(self, target: ResearchTarget) -> bool:
        return (
            self.proposal.target_id == target.target_id
            and self.proposal.display_name == target.display_name
            and self.proposal.source == target.source
            and self.proposal.evidence_checksum == target.evidence_checksum
        )


class TargetTransition(_TargetModel):
    """The run-level action required after a target decision."""

    action: Literal["no_change", "revise_run", "supersede_run"]
    target: ResearchTarget
    target_revision: int = Field(ge=0)
    parent_run_id: str | None = None
    reuse_prior_artifacts: bool = False
    approved: bool = False
    materialized: Literal[True] = True

    @field_validator("parent_run_id")
    @classmethod
    def validate_parent_run_id(cls, value: str | None) -> str | None:
        return None if value is None else _require_non_empty(value, "parent_run_id")

    @model_validator(mode="after")
    def validate_superseding_transition(self) -> "TargetTransition":
        if self.action == "supersede_run" and self.parent_run_id is None:
            raise ValueError("supersede_run requires parent_run_id")
        if self.action == "supersede_run" and self.reuse_prior_artifacts:
            raise ValueError("supersede_run cannot reuse prior artifacts")
        if self.action != "supersede_run" and self.parent_run_id is not None:
            raise ValueError("parent_run_id is only valid for supersede_run")
        return self


class TargetPolicyPreview(_TargetModel):
    """A non-persistable stage-policy result for a possible late target change."""

    action: Literal["supersede_preview"]
    target: ResearchTarget
    target_revision: int = Field(ge=0)
    parent_run_id: None = None
    reuse_prior_artifacts: Literal[False] = False
    approved: Literal[False] = False
    materialized: Literal[False] = False


_PRE_BUILD_STAGES = {"created", "reading", "feasibility_review"}
_LATE_STAGES = {"building", "verification", "version_archiving"}


def transition_target(
    current_target: ResearchTarget | str,
    proposal: TargetProposal | None = None,
    *,
    decision: TargetDecision | None = None,
    stage: str,
    parent_run_id: str | None = None,
    approved: bool | None = None,
) -> TargetTransition | TargetPolicyPreview:
    """Apply an approved proposal or classify a late-stage target change.

    Passing a string without a proposal is a lightweight stage-policy query kept
    for callers that only need to determine whether a new run is required.
    Materialized transitions always pass a proposal and human decision.
    """
    if stage in {"before_build", "pre_build"}:
        stage = "feasibility_review"
    if stage not in _PRE_BUILD_STAGES | _LATE_STAGES:
        raise ValueError(f"unknown workflow stage: {stage!r}")

    if approved is False:
        raise ValueError("an approved human target decision is required")

    if proposal is None:
        if stage in _LATE_STAGES:
            target = (
                current_target
                if isinstance(current_target, ResearchTarget)
                else ResearchTarget(
                    target_id=current_target,
                    display_name=current_target,
                    revision=0,
                    source="transition_query",
                    evidence_checksum="query",
                    approved=False,
                )
            )
            return TargetPolicyPreview(
                action="supersede_preview",
                target=target,
                target_revision=target.revision,
            )
        target = current_target if isinstance(current_target, ResearchTarget) else ResearchTarget.seed(current_target, current_target)
        return TargetTransition(
            action="no_change",
            target=target,
            target_revision=target.revision,
            parent_run_id=None,
            reuse_prior_artifacts=False,
            approved=target.approved,
        )

    if decision is None or not decision.approved:
        raise ValueError("an approved human target decision is required")
    if decision.proposal is None:
        raise ValueError("target decision must be bound to the supplied proposal")
    if decision.proposal is not proposal:
        raise ValueError("target decision is bound to a different proposal")
    if decision.target_id != proposal.target_id:
        raise ValueError("target decision does not match proposal")
    if decision.source != proposal.source:
        raise ValueError("target decision source does not match proposal")
    if decision.evidence_checksum != proposal.evidence_checksum:
        raise ValueError("target decision evidence_checksum does not match proposal")
    current = current_target if isinstance(current_target, ResearchTarget) else ResearchTarget.seed(current_target, current_target)
    if (
        proposal.target_id == current.target_id
        and proposal.display_name == current.display_name
        and proposal.source == current.source
        and proposal.evidence_checksum == current.evidence_checksum
        and current.approved
    ):
        return TargetTransition(
            action="no_change",
            target=current,
            target_revision=current.revision,
            parent_run_id=None,
            reuse_prior_artifacts=False,
            approved=True,
        )

    next_target = ResearchTarget.from_approval(
        proposal,
        decision,
        revision=current.revision + 1,
    )
    if stage in _LATE_STAGES:
        if parent_run_id is None:
            raise ValueError("supersede_run requires parent_run_id")
        return TargetTransition(
            action="supersede_run",
            target=next_target,
            target_revision=next_target.revision,
            parent_run_id=parent_run_id,
            reuse_prior_artifacts=False,
            approved=True,
        )
    return TargetTransition(
        action="revise_run",
        target=next_target,
        target_revision=next_target.revision,
        parent_run_id=None,
        reuse_prior_artifacts=False,
        approved=True,
    )


TargetApprovalReceipt.model_rebuild()
ResearchTarget.model_rebuild()
